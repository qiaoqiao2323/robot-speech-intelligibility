# Load necessary libraries
library(readxl)
library(dplyr)
library(ggplot2)

# Specify the file path
file_path <- "C:/Users/Administrator/OneDrive - UGent/Desktop/qq's document/vacabulary_game/results/E1/final.xlsx"


# Read the Excel file into a data frame
df <- read_excel(file_path, sheet = "Sheet1")
df$predict <- as.factor(df$predict)
df$annoylevel <- as.factor(df$annoylevel)


result2 <- df %>%
  group_by(predict) %>%
  summarize(
    score_mean = mean(score),
    score_std = sd(score),
    user_experience_mean = mean(user_experience),
    user_experience_std = sd(user_experience),
  )
result2



# Basic violin plot
p_scores <- ggplot(df, aes(x=predict, y=score,fill = predict)) + 
  geom_violin(trim = TRUE) +
  geom_boxplot(width = 0.1, fill = "white") + theme_minimal() +
  ylim(-0.2, 1.4) +
  stat_summary(fun.data=mean_sdl, mult=2, 
               geom="pointrange", color="red")+
  labs(title = "Scores")

p_scores




# Basic violin plot
p_user <- ggplot(df, aes(x=predict, y=user_experience, fill = predict)) + 
  geom_violin(draw_quantiles = c(0.5), color = "black") + # 设置中位线为黑色
  geom_violin(trim = TRUE) +
  geom_boxplot(width = 0.2, fill = "white") + theme_minimal() +
  ylim(-1, 12)+
  stat_summary(fun.data=mean_sdl,  
                      geom="pointrange", color="red")+
  labs(title = "user_experience")
  
p_user

p_scores <- ggplot(df, aes(x = predict, y = score, fill = predict, color = predict )) + 
  geom_boxplot(width = 0.1, fill = "white") + # Change to boxplot
  theme_minimal() +
  ylim(-1, 1.5) +
  stat_summary(fun.data = mean_sdl,  
               geom = "pointrange") +
  labs(title = "scores")

p_scores



p_user <- ggplot(df, aes(x = predict, y = user_experience, fill = predict, color = predict)) + 
  geom_boxplot(width = 0.1, fill = "white") + # Change to boxplot
  theme_minimal() +
  ylim(-2, 15) +
  stat_summary(fun.data = mean_sdl,  
               geom = "pointrange") +
  labs(title = "user_experience")

p_user



# 检验 score
wilcox_score <- wilcox.test(score ~ predict, data = df, paired = TRUE)
print(wilcox_score)

# 检验 user_experience
wilcox_user_experience <- wilcox.test(user_experience ~ predict, data = df, paired = TRUE)
print(wilcox_user_experience)



# Combine the plots
library(gridExtra)
grid.arrange(p_scores, p_user, ncol = 2)


# Filter the data for annoyance level 0 and create a violin plot
p_annoylevel_0 <- ggplot(df %>% filter(annoylevel == "0.97"), aes(x = predict, y = score, fill = predict)) + 
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white") + theme_minimal() +
  stat_summary(fun.data=mean_sdl, mult=1, 
               geom="pointrange", color="red")+
  ylim(-1, 2) +
  labs(title = "Annoylevel 0")




# Filter the data for annoyance level 1 and create a violin plot
p_annoylevel_1 <- ggplot(df %>% filter(annoylevel == "9.93"), aes(x = predict, y = score, fill = predict)) + 
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white") + theme_minimal() +
  ylim(-1, 2) +
  stat_summary(fun.data=mean_sdl, mult=1, 
               geom="pointrange", color="red")+
  labs(title = "Annoylevel 1")
p_annoylevel_1



# Combine the plots
library(gridExtra)
grid.arrange(p_annoylevel_0, p_annoylevel_1, ncol = 2)

