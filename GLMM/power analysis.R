library(readxl)
library(lme4)
library(glmnet)
library(ggplot2)
library(e1071)
library(scales)
library(simr)


file_path <- "C:/Users/Administrator/PycharmProjects/data_processing_word/merged_data_4.xlsx"
data <- read_excel(file_path, sheet = 'voice')
data_pilot <- read_excel(file_path, sheet = 'pilot')

data$user_experience <- factor(data$user_experience)

data$annoylevel <- rescale(data$annoylevel)
data$volume <- rescale(data$volume)
data$speed <- rescale(data$speed)
data$noisy <- rescale(data$noisy)
data$English <- rescale(data$English)
data$pitch <- rescale(data$pitch)
data$distance <- rescale(data$distance)
data$enunciation <- rescale(data$enuaciation)
data$room <- rescale(data$room)
data$score <- rescale(data$score)

fit_ux <- glmer(user_experience+0.0001 ~ room*volume+volume*noisy +annoylevel+ volume+ speed +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data,  family = Gamma(link = "log"))
fit_ps <- glmer(score+0.0001 ~  room*volume+ annoylevel+ volume+speed +noisy + English + pitch + distance + enuaciation + room + (1 | PN)+ (1 | word), data = data, family = Gamma(link = "log"))
summary(fit_ps)

library(sjPlot) #for plotting lmer and glmer mods

sjPlot::plot_model(fit_ps,show.values=TRUE, show.p=TRUE)

fit_ux <- glmer(user_experience ~ annoylevel+ volume+ speed + room*volume 
              +noisy+English + pitch + distance + enuaciation + (1 | PN), data = data, 
              family = Gamma(link = "log"),control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
summary(fit_ux)
sjPlot::plot_model(fit_ux,show.values=TRUE, show.p=TRUE)








fit_ux_1 <- clmm(user_experience ~ annoylevel+ volume+ speed+room*volume+noisy*volume+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, link = "logit")
results <- powerSim(fit_ps, test = fixed("annoylevel"), nsim = 20, progress = FALSE)
summary(fit_ps)

# post-hoc power analysis
power_results <- list()
# List of predictors - this includes the interaction term and the main effects
# Note: The interaction term in the model formula already includes the main effects, so they are not listed separately here
#interact_power <- powerSim(fit_ux, fcompare(~ annoylevel+ volume+speed +noisy + English + pitch + distance + enuaciation + room + (1 | PN)), nsim=20)
#print(interact_power)

predictors <- c("room:volume", "annoylevel", "speed", "room", "pitch")

# Loop over each fixed effect
for(predictor in predictors) {
  # Perform power analysis for the fixed effect
  fixef(fit_ps)['room:volume'] <- -0.6
  fixef(fit_ps)['pitch'] <- 0.6
  fixef(fit_ps)['room'] <- 0.6
  
  power_result <- powerSim(fit_ps, test = fixed(predictor), nsim = 1000)
  
  # Store the result
  power_results[[predictor]] <- power_result
  
  # Print the result
  print(power_result)
}
print(power_results)




fixef(fit_ps)['room:volume'] <- -0.9
# Set up the power analysis parameters for single predictors
powerAnalysis <- powerSim(fit_ps, test = fixed('room:volume'), nsim = 1000)
# Output the results
print(powerAnalysis)




#Prospective (A Priori) Power Analysis

pilotModel_ux <- glmer(user_experience+0.0001 ~ room*volume+volume*noisy +annoylevel+ volume+ speed +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data_pilot,  family = Gamma(link = "log"))
pilotModel_ps <- glmer(score+0.0001 ~  room*volume+ annoylevel+ volume+speed +noisy + English + pitch + distance + enuaciation+ room + (1 | PN), data = data_pilot, family = Gamma(link = "log"))


predictors <- c("room:volume", "annoylevel", "volume","noisy", "speed", "distance","room", "pitch","enuaciation")


# Loop over each fixed effect
for(predictor in predictors) {
  # Perform power analysis for the fixed effect
  power_result <- powerSim(pilotModel_ps, test = fixed(predictor), nsim = 20)
  
  # Store the result
  power_results[[predictor]] <- power_result
  
  # Print the result
  print(power_result)
}

#for clmm power analysis
library(ordinal)

# Your original model
fit_ux_1 <- clmm(user_experience ~ annoylevel + volume + speed + room*volume + noisy*volume + noisy + English + pitch + distance + enunciation + room + (1 | PN), data = data, link = "logit")

predictors <- c("annoylevel", "volume", "speed", "room", "noisy", "English", "pitch", "distance", "enunciation")
n_sim <- 20

for(pred in predictors) {
  significant_results <- numeric(n_sim)
  
  for(i in 1:n_sim) {
    # Simulate new dataset with only 'pred' varying
    simulated_data <- simulateDataForPredictor(fit_ux_1, pred)
    
    # Refit the model to the simulated data
    simulated_model <- clmm(user_experience ~ annoylevel + volume + speed + room*volume + noisy*volume + noisy + English + pitch + distance + enunciation + room + (1 | PN), data = simulated_data, link = "logit")
    
    # Check if the predictor 'pred' is significant
    significant_results[i] <- coef(summary(simulated_model))[pred, "Pr(>|z|)"] < 0.05
  }
  
  # Estimate the power for predictor 'pred'
  estimated_power <- mean(significant_results)
  cat("Estimated power for", pred, ":", estimated_power, "\n")
}


