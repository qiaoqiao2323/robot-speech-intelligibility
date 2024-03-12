library(readxl)
library(lme4)
library(glmnet)
library(ggplot2)
library(scales)
library(glmmTMB)
library(ordinal)
library(optimx)

install.packages("optimx")

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

data$annoylevel<- factor(data$annoylevel) #将变量设置为分类变量/虚拟变量
data$room<- factor(data$room) #将变量设置为分类变量/虚拟变量
data$noisy<- factor(data$noisy) #将变量设置为分类变量/虚拟变量
data$English<- factor(data$English) #将变量设置为分类变量/虚拟变量
data$user_experience <- factor(data$user_experience)

class(user_experience)
user_experience



# Fit a beta GLMM with random intercept and slope for time
model <- glmmTMB(score ~ annoylevel + volume + speed  + noisy + English + pitch + distance + enuaciation + room + (1 | PN), 
                 data = data, family = beta_family())

# Summarize the model
summary(model)

#data$Gender<- factor(data$Gender) #将变量设置为分类变量/虚拟变量
cc<- factor(data$gender) #将变量设置为分类变量/虚拟变量
class(cc)
print(cc)
# Fit a Poisson Generalized Linear Mixed Model (GLMM)
fit1 <- glmer(user_experience ~ annoylevel + volume + speed  + noisy + English + pitch + distance + enuaciation + room + (1 | PN), 
              data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ volume + speed + pitch + enuaciation  + (1 | PN), 
              data = data, family = poisson(link = "log"))

#volume

fit1 <- glmer(score+0.0001 ~ annoylevel+volume+speed+ I(annoylevel*volume)+ I(distance*volume) +I(room*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = Gamma(link = "identity"))

fit1 <- glmer(user_experience ~ annoylevel+volume+speed+ I(annoylevel*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+volume+speed+ I(distance*volume) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+volume+speed+ I(room*volume)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))

#speed
fit1 <- glmer(user_experience ~ annoylevel+volume+speed+ I(annoylevel*speed)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+volume+speed+ I(distance*speed) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+volume+speed+ I(room*speed)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))

#en
fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + I(annoylevel*enuaciation)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + I(distance*enuaciation) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + I(room*enuaciation) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))


#pitch
fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + I(annoylevel*pitch)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + I(distance*pitch) +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))
fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + I(room*pitch)+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, family = poisson(link = "log"))

fit1 <- clmm(user_experience ~ annoylevel+ volume+ speed+room*volume+noisy*volume+noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data, link = "logit")


fit1 <- glmer(user_experience ~ annoylevel+ volume+ speed + room*volume+room*noisy 
              +noisy+English + pitch + distance + enuaciation + (1 | PN), data = data, 
              family = Gamma(link = "log"),control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
#fit1 <- glmer(score+0.0001 ~ annoylevel+ volume+ speed +noisy+English + pitch + distance + enuaciation + room + (1 | PN), data = data,  family = Gamma(link = "log"))

summary_fit1 <- summary(fit1)
summary_fit1
library(sjPlot) #for plotting lmer and glmer mods

sjPlot::plot_model(fit1,show.values=TRUE, show.p=TRUE)

qqnorm(data$user_experience, main = "“Normal Q-Q plot of user experience”") 
qqline(data$user_experience)

hist(residuals(fit1), main =" “Histogram of residuals”", xlab = "“Residuals”")

qqnorm(data$score,  main = "“Normal Q-Q plot of user experience”") 
qqline(data$score)



summary(residuals)
qqnorm(residuals)
qqline(residuals)

residuals <- resid(fit1)
summary(residuals)
qqnorm(residuals)
qqline(residuals)




library(nortest)

ad.test(residuals)

ad.test(data$user_experience)
shapiro.test(residuals)


par(mfrow=c(2,2))
plot(fit1)

hist(data$user_experience, main = "Histogram of user experience", xlab = "User experience")



qqnorm(data$user_experience, main = "Normal Q-Q plot of user experience") 
qqline(data$user_experience)

hist(residuals(fit1), main = "“Histogram of residuals”", xlab = "“Residuals”")


qqnorm(residuals(fit1), main = "“Normal Q-Q plot of residuals”") 
qqline(residuals(fit1))


total_residuals

#fit1 <- lm(log(user_experience+0.0001) ~ annoylevel+ I((volume-1.5)^2) + I(speed^-1) + noisy+English + pitch + distance + enuaciation + room , data = data)

omcdiag(fit1, na.rm = TRUE, Inter = TRUE, detr = 0.01, red = 0.5)
imcdiag(fit1, na.rm = TRUE, Inter = TRUE, detr = 0.01, red = 0.5)
pcor(X, method = "pearson")
vif(fit1)
# Get the summary of the model
summary_fit1 <- summary(fit1)
summary_fit1

# Extract the t-values of the fixed effects
t_values <- summary_fit1$coefficients[, "t value"]

# Calculate the degrees of freedom
df <- nrow(data) - length(fixef(fit1))

# Calculate the p-values
p_values <- 2 * pt(abs(t_values), df, lower.tail = FALSE)

print(summary_fit1)
print(p_values)

# Add the p-values to the summary table
summary_fit1$coefficients <- cbind(summary_fit1$coefficients, "p value" = p_values)
print(summary_fit1)

# Calculate Cook's distance
cooks_d <- cooks.distance(fit1)

# Plot Cook's distance
plot(cooks_d, pch = 20, cex = 2, main = "Cook's Distance")
abline(h = 4 / nrow(data), col = "red", lty = 2)

par(mfrow = c(2, 2))
plot(fit1)

X <- data[, c("annoylevel", "volume", "speed", "pitch", "distance","gender", "noisy", "English", "enuaciation" ,"user_experience" )]
x <- data[, c("annoylevel", "volume", "speed", "pitch", "distance", "noisy", "English", "enuaciation" )]

Y <- data[, c("user_experience")]

cor2pcor(cov(X))
#cor.test(x, Y, method = c("pearson", "kendall", "spearman"))
cor(x, Y)

omcdiag(fit1, na.rm = TRUE, Inter = TRUE, detr = 0.01, red = 0.5)
imcdiag(fit1, na.rm = TRUE, Inter = TRUE, detr = 0.01, red = 0.5)
vif(fit1)

set.seed(123)
trainIndex <- sample(1:nrow(data), 0.8 * nrow(data))
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

#fit2 <- lmer(phonetic_similarity ~ annoylevel + volume + speed + pitch + enuaciation + (1 | PN), data = trainData)
fit2 <- lmer(log(phonetic_similarity + 0.0001) ~ annoylevel +gender+ volume + speed + noisy+English + pitch + distance + enuaciation + (1 | PN), data = data)

varmtx <- model.matrix(fit2)
response <- trainData$phonetic_similarity

cv.lasso <- cv.glmnet(varmtx, response, alpha = 1)
plot(cv.lasso)
abline(v = cv.lasso$lambda.min, col = "red", lty = 2)
abline(v = cv.lasso$lambda.1se, col = "blue", lty = 2)
best_lambda <- cv.lasso$lambda.min

best_model <- glmnet(varmtx, response, alpha = 0.5, lambda = best_lambda)
selectedVars <- names(best_model$lambda.min.var)[best_model$lambda.min.var != 0]
#testData$previous_var <- exp(testData$transformed_var)

predictions <- predict(fit2, newdata = testData)

RMSE <- sqrt(mean((predictions - testData$phonetic_similarity)^2))
MAE <- mean(abs(exp(predictions) - testData$phonetic_similarity))
R2 <- 1 - sum((predictions - testData$phonetic_similarity)^2) / sum((testData$phonetic_similarity - mean(testData$phonetic_similarity))^2)
print(MAE)
print(RMSE)
print(R2)




# Plot scatter plot of actual vs. predicted values
ggplot(data.frame(actual = testData$phonetic_similarity, predicted = predictions), aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  xlab("Actual Value") +
  ylab("Predicted Value") +
  ggtitle("Scatter Plot of Actual vs. Predicted Values")

