#' Prepare patient-level data for PU-learning and downstream modeling saving training and inference preprocessed data.
#'
#' @description
#' Processes longitudinal patient visit data to generate a patient-level dataset enriched with derived features,
#' scaled covariates, and PU-learning scores under the SAR assumption. It also creates training and inference datasets
#' with preprocessed data and saves them in Feather format for efficient downstream use.
#'
#' @param data A `data.table` or `data.frame` that must contain the following columns:
#'   - `patient_id`: Unique identifier for each patient (numeric).
#'   - `dead`: Binary indicator (0/1) for whether the patient is dead.
#'   - `death_time`: Time of death if it has occurred or censoring time otherwise (numeric).
#'   - `onset`: Binary indicator (0/1) for disease onset.
#'   - `onset_age`: Age at disease onset if it has occurred or `death_time` otherwise (numeric).
#'   - `age`: Patient's current age at that specific visit (numeric).
#'   - `visits`: Indicator of the current visit (numeric).
#'   The `data.frame` can contain extra columns with covariate values.
#'
#' @param cov_vector Character vector of covariate names to include in modeling. These will be scaled and encoded as needed.
#' @param lab_prop Numeric value (0–1) specifying the proportion of unlabeled patients to assign as negative in PU-learning thresholding.
#' @param train_path File path where the processed training dataset will be saved in Feather format.
#' @param infer_path File path where the processed inference dataset will be saved in Feather format.
#' @param pu_args Named list of PU-learning hyperparameters forwarded to \code{\link{pu_learning}}.

#'
#' @details
#' The function performs the following steps:
#' \enumerate{
#'   \item Feature engineering: Calculates hand crafted covariates.
#'   \item Scaling: Applies z-score normalization to numeric covariates.
#'   \item Encoding: Performs one-hot encoding of categorical variables.
#'   \item PU-learning: Estimates onset priors (\code{p_pos}) and assigns soft labels (\code{label_type}) based on \code{lab_prop}.
#'   \item Output preparation: Creates training and inference datasets and saves them as Feather files.
#' }
#'
#' @return A list with:
#'   \describe{
#'     \item{A processed data frame with patient-level information that will be used for fitting the multi-state models.}
#'     \item{Character vector of covariate names after one-hot encoding.}
#'   }
#'
#'Processed training and inference datasets are saved to disk at \code{train_path} and \code{infer_path}.

#' @examples
#' \dontrun{
#' prepare_data(
#'   data = panel_data,
#'   cov_vector = c("sex", "bmi", "smoking_status"),
#'   lab_prop = 0.2,
#'   train_path = "output/train.feather",
#'   infer_path = "output/infer.feather"
#' )
#' }
#'
#' @importFrom magrittr %>%
#' @importFrom dplyr arrange group_by summarize left_join mutate select slice ungroup
#' @importFrom dplyr if_else case_when row_number all_of desc first n
#' @importFrom stats quantile na.omit sd
#' @importFrom fastDummies dummy_cols
#' @importFrom arrow write_feather
#' @export


prepare_data <- function(data, cov_vector, lab_prop, pu_args, train_path, infer_path) {


  # --- Step 0: basic ordering
  scheme_visits <- data %>% arrange(patient_id, visits)

  # --- Step 1: last_bfo (last age before first onset per patient)
  # Find first onset visit index per patient
  first_onset <- scheme_visits %>%
    group_by(patient_id) %>%
    summarize(first_onset_idx = ifelse(any(onset == 1), min(which(onset == 1)), NA_integer_), .groups = "drop")

  # Last age before onset (NA if onset at first visit or no onset)
  last_bfo_df <- scheme_visits %>%
    left_join(first_onset, by = "patient_id") %>%
    group_by(patient_id) %>%
    mutate(last_bfo = dplyr::if_else(!is.na(first_onset_idx) & row_number() == first_onset_idx - 1,
                                     age, NA_real_)) %>%
    summarize(last_bfo = suppressWarnings(max(last_bfo, na.rm = TRUE)),
              .groups = "drop")
  last_bfo_df$last_bfo[!is.finite(last_bfo_df$last_bfo)] <- NA_real_

  # Mark patient-level onset (any visit with onset==1)
  has_onset <- scheme_visits %>%
    group_by(patient_id) %>%
    summarize(onset = as.integer(any(onset == 1)), .groups = "drop")

  # Last visit age per patient (for those without onset)
  last_visit <- data.table::as.data.table(scheme_visits)[, .SD[which.max(visits)], by = patient_id]
  last_visit <- as.data.frame(last_visit)[, c("patient_id", "age")]
  names(last_visit)[2] <- "last_visit_age"

  # Build per-patient table (first row per patient just as a carrier of features)
  scheme_data <- scheme_visits %>%
    group_by(patient_id) %>%
    slice(1) %>%
    ungroup() %>%
    select(-visits) %>%
    select(-onset)


  # Attach markers
  scheme_data <- scheme_data %>%
    left_join(has_onset, by = "patient_id") %>%
    left_join(last_bfo_df, by = "patient_id") %>%
    left_join(last_visit, by = "patient_id")


  # last_bfo: if no onset, use last visit age; else last age before onset (might be NA if onset at first visit)
  scheme_data <- scheme_data %>%
    mutate(last_bfo = ifelse(onset == 1, last_bfo, last_visit_age)) %>%
    select(-last_visit_age)

  out_scheme <- scheme_data

  # --- Step 2: Building variables related to follow-up process for PU-learning from full visit table
  # number of visits, length of the interval for possible disease development, frequency of visits
  len_fu <- scheme_visits %>%
    group_by(patient_id) %>%
    summarize(length_followup = first(death_time) - min(age, na.rm = TRUE),
              visits = n(),
              visit_rate = ifelse(length_followup!=0, visits/ length_followup, 0), .groups = "drop")

  scheme_data <- scheme_data %>%
    left_join(len_fu, by = "patient_id")

  scheme_data <- scheme_data %>%
    mutate(interval = abs(ifelse(onset == 1, onset_age, death_time) - last_bfo))

  features_prop <- c( "visits", "interval", "visit_rate")


  # --- Step 3: Scaling (z-score on numeric features cl and propensity for pu learning)
  is_binary01 <- function(x) {
    if (!is.numeric(x)) return(FALSE)
    ux <- unique(na.omit(x))
    length(ux) <= 2 && all(ux %in% c(0, 1))
  }
  scale_vec <- function(x) {
    mu <- mean(x, na.rm = TRUE)
    sdv <- sd(x, na.rm = TRUE)
    if (!is.finite(sdv) || sdv == 0) return(rep(0, length(x)))
    (x - mu) / sdv
  }

  num_covs <- cov_vector[
    vapply(scheme_data[cov_vector], is.numeric, TRUE) &
      !vapply(scheme_data[cov_vector], is_binary01, TRUE)
  ]

  scheme_data_scaled <- scheme_data
  for (c in c(num_covs,features_prop)) {
    if (c %in% names(scheme_data_scaled)) {
      scheme_data_scaled[[c]] <- scale_vec(scheme_data_scaled[[c]])
    }
  }

  # --- Step 4: Computing priors of disease onset with PU learning under SAR assumption

  features_cl   <- unique(c(cov_vector, "age"))
  features_prop <- c( "visits", "interval", "visit_rate")

  scores <- pu_learning(scheme_data_scaled, features_cl, features_prop, pu_args)

  zero_candidates <- scores %>% dplyr::filter(onset == 0) %>% arrange(desc(p_pos))
  thresh <- as.numeric(quantile(zero_candidates$p_pos, lab_prop, na.rm = TRUE))
  zero_onset_ids <- zero_candidates$patient_id[zero_candidates$p_pos < thresh]

  scores <- scores %>%
    mutate(label_type = case_when(
      onset == 1 ~ 1,
      onset == 0 & patient_id %in% zero_onset_ids ~ 0,
      TRUE ~ NA_real_
    )) %>%
    select(patient_id, label_type, p_pos)

  scheme_data_scaled <- scheme_data_scaled %>%
    left_join(scores, by = "patient_id")

  # --- Step 5: Preprocessing for the following phase: one-hot encoding

  factor_cols <- names(scheme_data_scaled)[
    sapply(scheme_data_scaled, is.factor)
  ]

  scheme_data_encoded <- if (length(factor_cols) > 0) {
    fastDummies::dummy_cols(
      scheme_data_scaled,
      select_columns = factor_cols,
      remove_first_dummy = TRUE,   # avoids multicollinearity
      remove_selected_columns = TRUE
    )
  } else {
    scheme_data_scaled
  }

  common_covs <- intersect(cov_vector, names(scheme_data_encoded))

  only_in_encoded <- setdiff(names(scheme_data_encoded), names(scheme_data_scaled))

  covariate_names_encoded <- unique(c(common_covs, only_in_encoded))


  # --- Step 5: Sacving Train and infer data
  training_data <- scheme_data_encoded %>%
    mutate(
      a = last_bfo,
      b = ifelse(onset == 1, onset_age, death_time),
      onset_prior = p_pos,
      onset_soft = label_type
    ) %>%
    select(patient_id, age, a, b, onset_soft, onset_age, onset_prior,
           dplyr::all_of(covariate_names_encoded))

  infer_data <- training_data %>%
    select(patient_id, age, a, b, onset_prior, dplyr::all_of(covariate_names_encoded))

  # --- Step 5: Save
  dir.create(dirname(train_path), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(infer_path), recursive = TRUE, showWarnings = FALSE)
  arrow::write_feather(training_data, train_path)
  arrow::write_feather(infer_data,   infer_path)

  # Clean return (without PU internals)
  out_scheme <- as.data.frame(out_scheme)
  return(list(out_scheme, covariate_names_encoded))
}
