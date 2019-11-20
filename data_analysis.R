library(e1071)
library(neuralnet) 
library(rsample)      
library(randomForest)
library(caret)  

setwd("/") #set working directory
dat1 <-read.csv("pm10.csv") #read data

row.number <- sample(1:nrow(dat1), 0.8*nrow(dat1))
train = dat1[row.number,]
test = dat1[-row.number,]
sqrt(mean(model1$residuals^2))
dim(train)
dim(test)
model1 = lm(log(S1)~GRIMM, data=train) #apply linear model
summary(model1)
pred1 <- predict(model1, newdata = test)
rmse <- sqrt(sum((exp(pred1) - test$S1)^2)/length(test$S1))
c(RMSE = rmse, R2=summary(model1)$r.squared)
results <- data.frame(Actual = test$S1, Prediction = pred1)
write.csv(results, file = "LR1.csv",row.names=FALSE)
#mydata <- dat1[-c(1, 2, 3), ]
par(mfrow=c(2,2))
plot(model1)
shapiro.test(model1$residuals)

#Implementing Support Vector Machine Model
model <- svm(S1 ~ GRIMM , train)
summary(model)
pred1 <- predict(model, test)
rmse <- sqrt(sum((exp(pred1) - test$S1)^2)/length(test$S1))
c(RMSE = rmse, R2=summary(model)$r.squared)
results <- data.frame(Actual = test$S1, Prediction = pred1)
write.csv(results, file = "SV1.csv",row.names=FALSE)

error <- mydata$S1.PM2.5 - predictedY
rmse <- function(errval)
{
  val = sqrt(mean(errval^2))
  return(val)
}
svrPredictionRMSE <- rmse(error)
svrPredictionRMSE 

# perform a grid search
tuneResult <- tune(svm, S1.PM2.5 ~ NG.PM2.5,  data = mydata,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
)
print(tuneResult)
# Visualize tuning graph
plot(tuneResult)

#If further tuning is needed
tuneResult <- tune(svm, S1.PM2.5 ~ NG.PM2.5,  data = mydata,
                   ranges = list(epsilon = seq(0,0.2,0.01), cost = 2^(2:9))
) 

print(tuneResult)
plot(tuneResult)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, mydata)
error <- mydata$NG.PM2.5 - tunedModelY
tunedModelRMSE <- rmse(error) 

#Implementing neural network model
mydata <- read.csv("nndata.csv")
scaleddata<-scale(mydata) #scale data and then normalize
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
maxmindf <- as.data.frame(lapply(mydata, normalize))
trainset <- maxmindf[1:160, ]
testset <- maxmindf[161:200, ]

nn <- neuralnet(S1.PM2.5 ~ NG.PM2.5,data=trainset, hidden=c(2,1), linear.output=TRUE, threshold=0.01)

nn$result.matrix
plot(nn) #Plot to see the neural network
temp_test <- subset(testset, select = c("NG.PM2.5"))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$S1.PM2.5, prediction = nn.results$net.result)
results

#Second NN approach
library(neuralnet)
mydata <- read.csv("ND1.csv")
index <- sample(1:nrow(mydata),round(0.80*nrow(mydata)))
train <- mydata[index,]
test <- mydata[-index,]
maxs <- apply(mydata, 2, max) 
mins <- apply(mydata, 2, min)
scaled <- as.data.frame(scale(mydata, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]
n <- names(train_)
f <- as.formula(paste("S1 ~", paste(n[!n %in% "S1"], collapse = " + ")))
 
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
#plot(nn)
pr.nn <- compute(nn,test_[,1:2])
pr.nn_ <- pr.nn$net.result*(max(mydata$S1)-min(mydata$S1))+min(mydata$S1)
test.r <- (test_$S1)*(max(mydata$S1)-min(mydata$S1))+min(mydata$S1)
 
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)

MSE.nn
sqrt(MSE.nn)
plot(test$S1.PM2.5,pr.nn_,col = c("red", "blue"),main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('Predicted','Observed'),pch=18,col=c('blue','red'), bty='y', cex = 0.75)
results <- data.frame(actual = test$S1, prediction = pr.nn_)
write.csv(results, file = "nn1.csv",row.names=FALSE)

# Random Forest Model implementation
dat1 <-read.csv("pm10.csv")
row.number <- sample(1:nrow(dat1), 0.8*nrow(dat1))
train = dat1[row.number,]
test = dat1[-row.number,]
m1 <- randomForest(
  formula = S1 ~ GRIMM,
  data    = train
)

m1

plot(m1)
which.min(m1$mse)
sqrt(m1$mse[which.min(m1$mse)]) #for training data

pred_randomForest <- predict(m1, test)
actual <- test$S1
predicted <- pred_randomForest
R2 <- 1 - (sum((actual-predicted)^2)/sum((actual-mean(actual))^2))
R2
caret::RMSE(predicted,actual)
results <- data.frame(Actual = test$S1, Prediction = predicted)
write.csv(results, file = "RF1.csv",row.names=FALSE)
