# ===================== Run over real data  ========================
library(dplyr)
library(devtools)
#panel_data <- readRDS("panel_data.RDS")
panel_data <- panel_data %>%
  filter(!is.na(onset))

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



estimates <- run_streams(
    panel_data,
    cov_vector,
    python =Sys.which("python3"),
    pu_args = list(verbose =TRUE)
)

### NEXT STEPS
# - functions to check the input dataframe and tips on how handle mising data   DONE
# - custom formula to design covariate effect in multi-state model              DONE
# - find a proper place to store intermediate results from python               DONE
# - fix main function with all arguments included                               DONE (but to be checked again)
# - checking documentation of each function
# - solving issues from check() function                                        DONE
# - store an example of dataset to run examples
# - build the flexsurv object as output                                         1
# - build summary function
# - test over snack dataset and compare with msm and flexsurv
# - provide logs plots for debugging
# - propagate seed








# Remark over Fixmatch consistency
#In our framework, the FixMatch component is not interpreted as a conventional loss to be minimized in isolation,
#but rather as a consistency constraint that regulates the agreement between student and teacher predictions on unlabeled data.
#Its role is to inject information from confident unlabeled samples while preserving the stability of the supervised and generative objectives.
#During training, the FixMatch weight is gradually increased through a ramp-up schedule, allowing the student to first stabilize on labeled data
#and reliable age supervision. After this saturation phase, a mild decay is applied to the FixMatch weight, ensuring that the consistency constraint
#remains active but does not dominate the optimization when its marginal contribution becomes smaller.
#As a result, the raw FixMatch discrepancy (i.e., the unweighted consistency error) continues to decrease over epochs,
#indicating that student and teacher predictions progressively align. At the same time, the weighted FixMatch contribution may increase during
#intermediate training stages due to the scheduled weight, without signaling optimization failure.
#Importantly, this behavior does not negatively affect either training or validation performance: the validation loss—computed without FixMatch and using the EMA
#teacher on unaugmented inputs—continues to improve or stabilize, demonstrating that the growing influence of the FixMatch term does not induce overfitting or degradation of generalization.
#This separation between training constraints and validation objectives highlights that FixMatch acts as a regularizing force shaping the learning dynamics,
#rather than as a direct target whose absolute magnitude must monotonically decrease.
