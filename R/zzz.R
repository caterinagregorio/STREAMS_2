.onLoad <- function(libname, pkgname) {
  utils::globalVariables(c(
    "patient_id", "visits", "onset", "first_onset_idx", "age",
    "last_bfo", "last_visit_age", "death_time", "length_followup",
    "onset_age", "p_pos", "label_type", "a", "b", "onset_soft",
    "onset_prior", "trans", ".SD"
  ))
}
