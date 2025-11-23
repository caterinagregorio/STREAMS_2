# ===================== Run over real data  ========================
library(dplyr)
library(devtools)
panel_data <- readRDS("panel_data.RDS")
panel_data <- panel_data %>%
  filter(!is.na(onset))

version_name <- "first_trial"
cov_vector <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia", "educ_el", "ALCO_CONSUMP", "R_SMOKE", "ws0", "drugs0",
                "dm_sex", "life_alone", "sei_long_cat", "fin_strain_early")

check_input_data(panel_data)

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

panel_data <- panel_data %>%
  mutate(across(all_of(cov_vector), impute_one))

check_input_data(panel_data)
