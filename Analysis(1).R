rm(list=ls())
df=read.csv("C:/Users/ankit/Downloads/apy.csv")
rice=df[df$Crop=="Rice",]
wb_rice=rice[rice$State=="West Bengal",]
head(wb_rice)
library(dplyr)
weighted_means <- wb_rice %>%
  group_by(Crop_Year, Season) %>%
  summarize(
    total_weighted_production = sum(Production * Area),
    total_area = sum(Area),
    weighted_mean_production = ifelse(total_area > 0, total_weighted_production / total_area, NA),
    .groups = "drop"
  )
wb_rice_means=cbind(weighted_means$Crop_Year,weighted_means$Season,weighted_means$weighted_mean_production)
k=1:length(wb_rice_means)
wb_rice_means=cbind(wb_rice_means,k)
colnames(wb_rice_means)=c("Crop_Year","Season","W_Production","Rounds")
wb_rice_means=as.data.frame(wb_rice_means)
write.csv(up_wheat_mean,"C:/Users/ankit/Downloads/wb_rice_production.csv")


rm(list=ls())
df=read.csv("C:/Users/ankit/Downloads/apy.csv")
jute=df[df$Crop=="Jute",]
wb_jute=jute[jute$State=="West Bengal",]
library(dplyr)
weighted_means <- wb_jute %>%
  group_by(Crop_Year) %>%
  summarize(
    total_weighted_production = sum(Production * Area),
    total_area = sum(Area),
    weighted_mean_production = ifelse(total_area > 0, total_weighted_production / total_area, NA)
  )
wb_jute_means=cbind(weighted_means$Crop_Year,weighted_means$weighted_mean_production)
wb_jute_means=as.data.frame(wb_jute_means)
colnames(wb_jute_means)=c("Crop_Year","Weighted Means")
write.csv(wb_jute_means,"C:/Users/ankit/Downloads/wb_jute_production(1).csv")
