#' BISOGNA CONTROLLARE CON ATTENZIONE TUTTA FUNZIONE E SCRIVERE BENE DESCRIZIONE
#' Prepare per-patient training and inference data for trajectory reconstruction
#'
#' This function takes visit-level longitudinal data and constructs per-patient
#' datasets ready to be used for semi-supervised trajectory reconstruction
#' \itemize{
#'   \item orders visits by \code{patient_id} and \code{visits};
#'   \item computes the last age before first onset (\code{last_bfo}) and
#'         patient-level onset status;
#'   \item derives follow-up summaries such as \code{length_followup},
#'         \code{visits} and \code{visit_rate};
#'   \item builds a per-patient table and a follow-up interval
#'         (\code{interval}) between \code{last_bfo} and
#'         onset or censoring time;
#'   \item scales non-binary numeric covariates with z-scores;
#'   \item runs PU learning via \code{\link{pu_learning}} to obtain
#'         \code{p_pos} (prior onset probability) and a semi-supervised
#'         label \code{label_type};
#'   \item dummy-encodes factor covariates using
#'         \code{fastDummies::dummy_cols()}, adding the newly created
#'         dummy variables to the original \code{covariate_names};
#'   \item builds and saves a training table and an inference table as
#'         Feather files.
#' }
#'
#' @param data A `data.table` or `data.frame` that must contain the following columns with no missing values:
#'   - `patient_id`: Unique identifier for each patient (numeric).
#'   - `dead`: Binary indicator (0/1) for whether the patient is dead.
#'   - `death_time`: Time of death if it has occurred or censoring time otherwise (numeric).
#'   - `onset`: Binary indicator (0/1) for disease onset.
#'   - `onset_age`: Age at disease onset if it has occurred or `death_time` otherwise (numeric).
#'   - `age`: Patient's current age at that specific visit (numeric).
#'   - `visits`: Indicator of the current visit (numeric).
#'   The `data.frame` can contain extra columns with covariate values.
#'
#' @param cov_vector Character vector with the names of covariates to be used
#'   as inputs for the PU-learning step. These variables are taken from
#'   \code{scheme_data} (after scaling of non-binary numeric covariates).
#'
#' @param covariate_names Character vector with the names of covariates that
#'   should enter the final training and inference tables. After dummy
#'   encoding, the function constructs \code{covariate_names_encoded} by
#'   keeping:
#'   \itemize{
#'     \item those covariates in \code{covariate_names} that still exist in
#'           \code{scheme_data_encoded};
#'     \item plus all columns that are present only in the encoded data
#'           (typically the dummy variables created from factor covariates).
#'   }
#' @param lab_prop Numeric scalar in \eqn{(0, 1)} indicating the proportion of
#'   patients with \code{onset == 0} to be treated as reliable negatives in the
#'   PU-learning step. Patients below the corresponding quantile of
#'   \code{p_pos} are assigned \code{label_type = 0}, while patients with
#'   \code{onset == 1} are assigned \code{label_type = 1}, and the remaining
#'   patients receive \code{NA}.
#' @param train_path Character scalar giving the file path where the per-patient
#'   training dataset should be saved as a Feather file
#'   (via \code{arrow::write_feather()}).
#' @param infer_path Character scalar giving the file path where the per-patient
#'   inference dataset should be saved as a Feather file.
#'
#' @details
#' The function first aggregates visit-level data to patient-level summaries.
#' \code{last_bfo} is defined as the last observed age before first onset; if a
#' patient never experiences onset, \code{last_bfo} is set to the age at the
#' last visit. The follow-up interval \code{interval} is computed as the
#' absolute difference between \code{last_bfo} and the event time
#' (\code{onset_age} if \code{onset == 1}, or \code{death_time} otherwise).
#'
#' PU learning is performed by \code{\link{pu_learning}}, which is expected to
#' return at least the columns \code{patient_id}, \code{onset} and \code{p_pos}.
#' From these, the function constructs a semi-supervised label
#' \code{label_type} based on \code{lab_prop}.
#'
#' Factor covariates in \code{scheme_data_scaled} are dummy-encoded using
#' \code{fastDummies::dummy_cols()} with \code{remove_first_dummy = TRUE} and
#' \code{remove_selected_columns = TRUE}, so that the original factor columns
#' are removed and replaced by (k-1) dummy variables.
#'
#' The training dataset contains, for each patient:
#' \itemize{
#'   \item \code{patient_id}, \code{age};
#'   \item \code{a}: \code{last_bfo};
#'   \item \code{b}: \code{onset_age} if \code{onset == 1}, otherwise
#'         \code{death_time};
#'   \item \code{onset_soft}: the semi-supervised label \code{label_type};
#'   \item \code{onset_age}, \code{onset_prior} (\code{p_pos});
#'   \item all covariates in \code{covariate_names_encoded}.
#' }
#'
#' The inference dataset is a reduced version of the training dataset that drops
#' \code{onset_soft} and \code{onset_age}, and keeps
#' \code{patient_id}, \code{age}, \code{a}, \code{b}, \code{onset_prior} and the
#' encoded covariates.
#'
#' @return
#' (To be aligned with the function body.)
#' Currently the function writes two Feather files to \code{train_path} and
#' \code{infer_path}. You may want to make the function return, for example,
#' a list such as:
#' \code{list(training_data = training_data, infer_data = infer_data,
#' covariate_names_encoded = covariate_names_encoded)}.
#'
#' @seealso \code{\link{pu_learning}}, \code{fastDummies::dummy_cols()},
#'   \code{arrow::write_feather()}
#'
#' @examples
#' \dontrun{
#' train_file  <- "data/processed/train.feather"
#' infer_file  <- "data/processed/infer.feather"
#'
#' cov_vector       <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia")
#' covariate_names  <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia",
#'                       "educ_el", "ALCO_CONSUMP")
#'
#' out <- prepare_data(
#'   data            = visits_data,
#'   cov_vector      = cov_vector,
#'   covariate_names = covariate_names,
#'   lab_prop        = 0.2,
#'   train_path      = train_file,
#'   infer_path      = infer_file
#' )
#' }

prepare_data <- function(data, cov_vector, covariate_names, lab_prop, train_path, infer_path) {


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
  last_visit <- as.data.table(scheme_visits)[, .SD[which.max(visits)], by = patient_id]
  last_visit <- as.data.frame(last_visit)[, c("patient_id", "age")]
  names(last_visit)[2] <- "last_visit_age"

  # Build per-patient table (first row per patient just as a carrier of features)
  scheme_data <- scheme_visits %>%
    group_by(patient_id) %>%
    slice(1) %>%
    ungroup() %>%
    select(-visits) %>%
    select(-onset)

  out_scheme <- scheme_data

  # Attach markers
  scheme_data <- scheme_data %>%
    left_join(has_onset, by = "patient_id") %>%
    left_join(last_bfo_df, by = "patient_id") %>%
    left_join(last_visit, by = "patient_id")


  # last_bfo: if no onset, use last visit age; else last age before onset (might be NA if onset at first visit)
  scheme_data <- scheme_data %>%
    mutate(last_bfo = ifelse(onset == 1, last_bfo, last_visit_age)) %>%
    select(-last_visit_age)


  # Adding variables for PU length_followup from full visit table
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

  # --- Step 2: Scaling (z-score on non-binary numeric covs)
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

  # --- Step 3: Priors (PU)

  features_cl   <- unique(c(cov_vector, "age"))
  features_prop <- c( "visits", "interval", "visit_rate")


  # sar_em_pu must return columns: patient_id, onset (0/1), p_pos
  scores <- pu_learning(scheme_data_scaled, features_cl, features_prop)

  zero_candidates <- scores %>% filter(onset == 0) %>% arrange(desc(p_pos))
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

  scheme_data_encoded <- fastDummies::dummy_cols(
    scheme_data_scaled,
    select_columns = names(Filter(is.factor, scheme_data_scaled)),
    remove_first_dummy = TRUE,   # avoids multicollinearity
    remove_selected_columns = TRUE
  )
  common_covs <- intersect(covariate_names, names(scheme_data_encoded))


  only_in_encoded <- setdiff(names(scheme_data_encoded), names(scheme_data_scaled))

  covariate_names_encoded <- unique(c(common_covs, only_in_encoded))


  # --- Step 4: Train + infer tables
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
  return(out_scheme)
}
