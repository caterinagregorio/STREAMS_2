# ===================== Run over real data  ========================
library(devtools)
#panel_data <- readRDS("panel_data.RDS")
panel_data <- panel_data %>%
  filter(!is.na(onset))

cov_vector <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia", "educ_el", "ALCO_CONSUMP", "R_SMOKE", "ws0", "drugs0",
                "dm_sex", "life_alone", "sei_long_cat", "fin_strain_early")

#check_input_data(panel_data)

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

t0 <- Sys.time()
fit_streams <- run_streams(
    data = panel_data,
    cov_vector = cov_vector,
    python = Sys.which("python3"),
    pu_args = list(verbose = TRUE)
)
t1 <- Sys.time()
time_streams <- as.numeric(difftime(t1, t0, units = "secs"))

lapply(fit_streams, function(x) summary(x, coefs = TRUE))

#trying all methods for my class flexsurvreg_pooled
trans1_streams <- fit_streams[[1]]
class(trans1_streams)
print(trans1_streams)
summary(trans1_streams, coefs = TRUE) # al momento è uguale al print ma si puo arricchire
confint(trans1_streams)
attr(trans1_streams, "rubin")



# COMPARING WITH MSM

prepare_msm<- function(df){
  panel_data_death <- df%>% filter(dead==1 ) %>% group_by(patient_id) %>% mutate(age=death_time) %>% distinct(patient_id,.keep_all = T)
  df2 <- rbind(df,panel_data_death) %>% group_by(patient_id) %>%  mutate(state=case_when(max(onset)==1 & onset_age<=age& age!=death_time~2 ,
                                                                                         death_time==age & dead==1 ~3,
                                                                                         TRUE~1)) %>% arrange(patient_id)
  return(df2)
}

msm_main <- function(df, cov_vector) {

  df <- prepare_msm(df)
  Q <- rbind(c(0, 1, 1),
             c(0, 0, 1),
             c(0, 0, 0))

  cov_vector <- intersect(cov_vector, names(df))

  cov_formula <- if (length(cov_vector) > 0) {
    stats::as.formula(paste("~", paste(cov_vector, collapse = " + ")))
  } else {
    ~ 1
  }

  fit <- tryCatch(
    msm::msm(
      state ~ age,
      subject    = patient_id,
      data       = df,
      qmatrix    = Q,
      covariates = cov_formula,
      censor     = 99,
      control    = list(fnscale = 1000, maxit = 1000),
      deathexact = TRUE
    ),
    error = function(e) NULL
  )

  coln <- c("rate", cov_vector)
  mat <- matrix(NA_real_, nrow = 3, ncol = length(coln),
                dimnames = list(c("0->1", "0->2", "1->2"), coln))

  if (is.null(fit)) {
    return(list(model = NULL, params = mat))
  }


  v <- fit$estimates
  vn <- names(v)

  # baseline intensities for 3 transitions then cov effects per transition
  needed <- 3 + 3 * length(cov_vector)
  vv <- rep(NA_real_, needed)
  vv[seq_len(min(length(v), needed))] <- v[seq_len(min(length(v), needed))]

  mat[, "rate"] <- vv[1:3]
  if (length(cov_vector) > 0) {
    off <- 4
    for (cv in cov_vector) {
      mat[, cv] <- vv[off:(off + 2)]
      off <- off + 3
    }
  }
  list(model = fit, params = mat)
}

cov_vector <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia", "educ_el", "ALCO_CONSUMP", "R_SMOKE", "ws0", "drugs0",
                "dm_sex", "life_alone", "sei_long_cat", "fin_strain_early")

t0 <- Sys.time()
fits_msm <-   msm_main(panel_data, cov_vector)
t1 <- Sys.time()
time_msm <- as.numeric(difftime(t1, t0, units = "secs") )




### NEXT STEPS
# - functions to check the input dataframe and tips on how handle mising data   DONE
# - custom formula to design covariate effect in multi-state model              DONE
# - find a proper place to store intermediate results from python               DONE
# - fix main function with all arguments included                               DONE (but to be checked again)
# - checking documentation of each function
# - solving issues from check() function                                        DONE
# - store an example of dataset to run examples
# - build the flexsurv object as output                                         DONE (to be checked)
# - build summary function                                                      DONE (to be checked)
# - test over snack dataset and compare with msm and flexsurv
# - provide logs plots for debugging
# - propagate seed
# - Metadata: attr(pooled_fit,"streams") to set info like logs as objects attributes
# - Set messages in main function to update user
# - Disclaimer on output for user??
# - Remove warnings() from pu_learning





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
