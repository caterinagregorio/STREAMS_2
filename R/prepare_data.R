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
    select(-visits)

  # Attach markers
  scheme_data <- scheme_data %>%
    select(-onset) %>%
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

  # Imputation
  impute_one <- function(x) {
    if (is.numeric(x)) {
      x[is.na(x)] <- mean(x, na.rm = TRUE)
      return(x)
    }
    if (is.factor(x) || is.character(x)) {
      mode_val <- names(sort(table(x), decreasing = TRUE))[1]
      x[is.na(x)] <- mode_val
      return(x)
    }
    x
  }

  scheme_data <- scheme_data %>%
    mutate(across(all_of(cov_vector), impute_one))

  out_scheme <- scheme_data

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
  for (c in num_covs) {
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



  # --- Step 4: Train + infer tables
  training_data <- scheme_data_scaled %>%
    mutate(
      a = last_bfo,
      b = ifelse(onset == 1, onset_age, death_time),
      onset_prior = p_pos,
      onset_soft = label_type
    ) %>%
    select(patient_id, age, a, b, onset_soft, onset_age, onset_prior,
           dplyr::all_of(covariate_names))

  infer_data <- training_data %>%
    select(patient_id, age, a, b, onset_prior, dplyr::all_of(covariate_names))

  # --- Step 5: Save
  dir.create(dirname(train_path), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(infer_path), recursive = TRUE, showWarnings = FALSE)
  arrow::write_feather(training_data, train_path)
  arrow::write_feather(infer_data,   infer_path)

  # Clean return (without PU internals)
  out_scheme <- as.data.frame(out_scheme)
  return(out_scheme)
}
