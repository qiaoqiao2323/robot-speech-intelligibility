library(readxl)
library(lme4)
library(glmnet)
library(ggplot2)
library(e1071)
library(scales)



file_path <- "C:/Users/Administrator/PycharmProjects/data_processing_word/merged_data_4.xlsx"
data <- read_excel(file_path, sheet = 'voice')


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




# Fit the linear mixed-effects model with normalized variables

fit1 <- lmer(score ~ annoylevel + volume + speed + noisy + English + pitch + distance + enuaciation + room +(1 | PN), data = data)

fit1 <- lmer(score ~ annoylevel + volume + speed + noisy + English + pitch + distance + enuaciation + room +(1 | PN), data = data)
fit1 <- lmer(score ~ annoylevel + noisy + English + pitch + distance + room + I(annoylevel*room)+ (1 | PN), data = normalized_data)



fit1 <- lm(log(score + 0.0001) ~ annoylevel + volume + speed + noisy+English + pitch + distance + enuaciation + room , data = data)
fit1 <- lm(log(user_experience + 0.0001) ~ annoylevel + volume + speed + noisy+English + pitch + distance + enuaciation + room , data = data)


library(lme4)
library(simr)

fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))


par(mfrow=c(2,2))
plot(fit1)



# Assuming 'fit1' is your LMM model object

# Calculate residuals


# install and load moments package
library(moments)

# generate some fake data
y <- data$user_experience # lognormal data
sum(is.na(y))

skewness(y)
x <- data$score # lognormal data
sum(is.na(x))


skewness_value <- skewness(1/(data$score+0.00001))
kurtosis_value <- kurtosis(data$score)


print(paste("Skewness:", skewness_value))
print(paste("Kurtosis:", kurtosis_value))
library(brms)
library(pscl)
library(Rfast2)
library(rjags)

install.packages("rjags")
library(zoib)

fit1 <- zoib(score ~ annoylevel + volume + speed + noisy + English + pitch + distance + enuaciation + room
             + 1 | PN, data = data, 
              random = 1, EUID= AlcoholUse$County,
             zero.inflation = TRUE, one.inflation = FALSE, joint = FALSE)

fit1 <- glmer(score+0.0001 ~  annoylevel + noisy + English + distance + room + (1 | PN), 
              data = data,
              family = Gamma(link = "identity"))

fit1 <- glmer(score+0.0001 ~  volume + speed + pitch  + enuaciation + (1 | PN), 
              data = data,
              family = Gamma(link = "log"))


#volume
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(annoylevel*volume)+ I(distance*volume) +I(room*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "identity"))
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(annoylevel*volume)+ I(distance*volume) +I(room*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "identity"))

fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(annoylevel*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(distance*volume) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(room*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))

#speed
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(annoylevel*speed)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(distance*speed) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(room*speed)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))

#en
fit1 <- glmer(score+0.0001 ~ annoylevel+ volume+ speed + I(annoylevel*enuaciation)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+ volume+ speed + I(distance*enuaciation) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+ volume+ speed + I(room*enuaciation) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))


#pitch
fit1 <- glmer(score+0.0001 ~ annoylevel+ volume+ speed + I(annoylevel*pitch)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+ volume+ speed + I(distance*pitch) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "log"))
fit1 <- glmer(score+0.0001 ~ annoylevel+volume+ speed +noisy+English + pitch + distance + enuaciation +(1 | PN), data = data, family = Gamma(link = "identity"))

library(sjPlot) #for plotting lmer and glmer mods
fit1 <- glmer(score+0.0001 ~  I(room*volume*annoylevel)+annoylevel+ volume+speed +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "inverse"))


fit1 <- glmer(score+0.0001 ~  I(room*volume)+annoylevel+ volume+speed +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "inverse"))

fit1 <- glmer(score+0.0001 ~  I(room*volume)+I(volume*annoylevel)+I(volume*distance)+annoylevel+volume+annoylevel+speed +noisy + pitch + room + (1 | PN), data = data, family = Gamma(link = "inverse"))



# Get the summary of the model
summary_fit1 <- summary(fit1)
summary_fit1



sjPlot::plot_model(fit1,show.values=TRUE, show.p=TRUE)



# Extract the t-values of the fixed effects
t_values <- summary_fit1$coefficients[, "t value"]
t_values
# Calculate the degrees of freedom
df <- nrow(data) - length(fixef(fit1))

# Calculate the p-values
p_values <- 2 * pt(abs(t_values), df, lower.tail = FALSE)

print(summary_fit1)




library(nortest)

ad.test(log(data$user_experience + 0.0001))
ad.test(log(data$score + 0.0001))

ad.test(data$user_experience)

residuals <- resid(fit1)
summary(residuals)
qqnorm(residuals)
qqline(residuals)
library(nortest)

ad.test(residuals)


print(p_values)
shapiro.test(resid(fit1))

# Add the p-values to the summary table
summary_fit1$coefficients <- cbind(summary_fit1$coefficients, "p value" = p_values)
# Calculate Cook's distance
cooks_d <- cooks.distance(fit1)

# Plot Cook's distance
plot(cooks_d, pch = 20, cex = 2, main = "Cook's Distance")
abline(h = 4 / nrow(data), col = "red", lty = 2)

par(mfrow = c(2, 2))
plot(fit1)

library(corpcor)
library(ppcor)
library(mctest)
library(car)

X <- data[, c("annoylevel", "volume", "speed", "pitch", "distance","gender", "noisy", "English", "enuaciation" ,"score" )]
cor2pcor(cov(X))

cor_matrix <- cor(X)
pcor_matrix <- pcor(X)
omcdiag(fit1, na.rm = TRUE, Inter = TRUE, detr = 0.01, red = 0.5)
imcdiag(fit1, na.rm = TRUE, Inter = TRUE, detr = 0.01, red = 0.5)
vif(fit1)

set.seed(123)
trainIndex <- sample(1:nrow(data), 0.8 * nrow(data))
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

#fit2 <- lmer(score ~ annoylevel + volume + speed + pitch + enuaciation + (1 | PN), data = trainData)
fit2 <- lmer(log(score + 0.0001) ~ annoylevel +gender+ volume + speed + noisy+English + pitch + distance + enuaciation + (1 | PN), data = data)

varmtx <- model.matrix(fit1)
response <- trainData$score

cv.lasso <- cv.glmnet(varmtx, response, alpha = 1)
plot(cv.lasso)
abline(v = cv.lasso$lambda.min, col = "red", lty = 2)
abline(v = cv.lasso$lambda.1se, col = "blue", lty = 2)
best_lambda <- cv.lasso$lambda.min

best_model <- glmnet(varmtx, response, alpha = 0.5, lambda = best_lambda)
selectedVars <- names(best_model$lambda.min.var)[best_model$lambda.min.var != 0]
#testData$previous_var <- exp(testData$transformed_var)

predictions <- predict(fit1, newdata = testData)

RMSE <- sqrt(mean((predictions - testData$score)^2))
MAE <- mean(abs(exp(predictions) - testData$score))
R2 <- 1 - sum((predictions - testData$score)^2) / sum((testData$score - mean(testData$score))^2)
print(MAE)
print(RMSE)
print(R2)




# Plot scatter plot of actual vs. predicted values
ggplot(data.frame(actual = testData$score, predicted = predictions), aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  xlab("Actual Value") +
  ylab("Predicted Value") +
  ggtitle("Scatter Plot of Actual vs. Predicted Values")

