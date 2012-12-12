library(xtable)
library(beanplot)

#### Constants

# Stem/Tier
expNum = 'sim1'
trfile = 'Dataframes/StemTier9-25training.csv'
tefile = 'Dataframes/StemTier9-25testing.csv'
confile = 'Dataframes/StemTier9-25constraints.csv'
tcolnames = c('Errors', 'Stem', 'Tier', 'Iteration')
tcolclasses = c('numeric', 'factor', 'numeric', 'integer')
ccolnames = c('Con', 'Stem', 'Tier', 'Iteration')
ccolclasses = c('integer', 'factor', 'numeric', 'integer')
labels = c('F/0', 'F/50', 'F/90', 'T/0', 'T/50', 'T/90')
trainingSize = 3829 # exceptions removed
#testingSize = 983 # exceptions in
testingSize = 957
sizes = c(trainingSize, testingSize, 1)
rates = c(100, 100, 1)

# LR/IF
# expNum = 'sim2'
# trfile = 'DetGenLR/training/detgenlrtraining.csv'
# tefile = 'DetGenLR/testing/detgenlrtesting.csv'
# confile = 'DetGenLR/constraints/detgenlrconstraints.csv'
# tcolnames = c('Errors', 'LearningRate', 'InductionFreq', 'Iteration')
# tcolclasses = c('numeric', 'factor', 'factor', 'integer')
# ccolnames = c('Con', 'LearningRate', 'InductionFreq', 'Iteration')
# ccolclasses = c('integer', 'factor', 'factor', 'integer')
# labels = c('.01/10', '.1/10', '.3/10', '.01/50', '.1/50', '.3/50', '.01/90',
#             '.1/90', '.3/90')

#### Functions

getData = function(filename, colnames, colclasses, iternum) {
  newdata = read.csv(filename, header = FALSE, sep = ',', 
                     col.names = colnames, colClasses = colclasses)
  newdata = newdata[newdata$Iteration == iternum,]
  return(newdata)
}

printTable = function(data, func, n){
  print(xtable(tapply(data[,1] * rates[n], list(data[,2], data[,3]), func)))
}

printMeanSd = function(data){
  printTable(data, mean, n)
  printTable(data, sd, n)
}

meanAndSd = function(data){
  return(paste(round(mean(data),3), " (", round(sd(data),3), ")", sep=''))
}

plotBeans = function(data, type){
  filename = paste(c(expNum, type, 'bean', '.jpg'))
  dv = ''
  if (type != 'Constraints'){
    dv = ' Error Rate'
  }
  #jpeg(filename)
  beanplot(data[,1] ~ data[,2] + data[,3], main = paste(c(type, dv)), 
           xlab = 'Conditions (Stem Constraints/Tier Frequency)', 
           ylab = dv, log = '', names = labels, cex.axis = .9, cex.lab = .95)
  #dev.off()
}

printReg = function(regression, title){
  print(xtable(summary(regression), label = title, digits =
    c(0,3,3,3,3)), table.placement = 'tbp', caption.placement = 'top')
}

regStraight = function(data, n){
  glm(data[,1] * sizes[n] ~ data[,2] + data[,3], family = "poisson")
}

regInt = function(data, n){
  glm(data[,1] * sizes[n] ~ data[,2] * data[,3], family = "poisson")
}


### Application

# read data
tr = getData(trfile, tcolnames, tcolclasses, 4)
#tr$Errors = 100 * tr$Errors
#tr$Errors = trainingSize * tr$Errors
summary(tr)

#tr$Tier = factor(tr$Tier, levels = c(0, 0.5, 0.9))

te = getData(tefile, tcolnames, tcolclasses, 4)
#te$Errors = 100 * te$Errors
#te$Errors = testingSize * te$Errors

te$Tier = as.factor(te$Tier)

con = getData(confile, ccolnames, ccolclasses, 4)

con$Tier = as.factor(con$Tier)

# print tables
printMeanSd(tr, 1)
printMeanSd(te, 2)
printMeanSd(con, 3)

# make beanplots
plotBeans(tr, 'Training')
plotBeans(te, 'Testing')
plotBeans(con, 'Constraints')

# regression
trreg1 = regStraight(tr, 1)
summary(trreg1)
printReg(trreg1, 'Regression on Error Rate in First Training')
trreg2 = regInt(tr, 1)
summary(trreg2)
printReg(trreg2, 'Regression on Error Rate in First Training')
anova(trreg1, trreg2, test="Chisq")
                     
tereg1 = regStraight(te, 2)
summary(tereg1)
printReg(tereg1, 'Regression on Error Rate in Last Testing')
tereg2 = regInt(te, 2)
summary(tereg2)
printReg(tereg2, 'Regression on Error Rate in Last Testing')
anova(tereg1, tereg2, test="Chisq")

conreg1 = regStraight(con, 3)
summary(conreg1)
printReg(conreg1, 'Regression on Final Number of Constraints')
conreg2 = regInt(con, 3)
summary(conreg2)
printReg(conreg2, 'Regression on Final Number of Constraints')
anova(conreg1, conreg2, test="Chisq")

# interaction plots
jpeg('exp5interactionstemtier.jpg')
interaction.plot(training[training$Iteration == 0,]$Stem,
		 training[training$Iteration == 0,]$Tier,
		 training[training$Iteration == 0,]$Errors, fun = mean, 
		 xlab = 'Stem', ylab = 'Errors in First Training', main =
		 'Interaction in Training Errors',
		 trace.label = 'Tier Freq.')
dev.off()
jpeg('exp5interactiontierstem.jpg')
interaction.plot(training[training$Iteration == 0,]$Tier,
		 training[training$Iteration == 0,]$Stem,
		 training[training$Iteration == 0,]$Errors, fun = mean, 
		 trace.label = 'Stem', ylab = 'Errors in First Training', main =
		 'Interaction in Training Errors',
		 xlab = 'Tier Frequency')

dev.off()

jpeg('stem-tier-test-interaction.jpg')
interaction.plot(te$Stem,
		 te$Tier,
		 te$Errors, fun = mean, 
		 xlab = 'Stem', ylab = 'Errors in Last Testing', main =
		 'Interaction in Testing Errors',
		 trace.label = 'Tier Freq.')
dev.off()
jpeg('tier-stem-test-interaction.jpg')
interaction.plot(te$Tier,
		 te$Stem,
		 te$Errors, fun = mean, 
		 trace.label = 'Stem', ylab = 'Errors in Last Testing', main =
		 'Interaction in Testing Errors',
		 xlab = 'Tier Frequency')
dev.off()
jpeg('stem-tier-con-interaction.jpg')
interaction.plot(con$Stem,
                 con$Tier,
                 con$Con, fun = mean, 
                 xlab = 'Stem', ylab = 'Number of Constraints', main =
                   'Interaction in Number of Constraints',
                 trace.label = 'Tier Freq.')
dev.off()

jpeg('tier-stem-con-interaction.jpg')
interaction.plot(constraints$Tier,
                 constraints$Stem,
                 constraints$Con, fun = mean, 
                 trace.label = 'Stem', ylab = 'Number of Constraints', main =
                   'Interaction in Number of Constraints',
                 xlab = 'Tier Frequency')
dev.off()


# testing without outliers

boxplot.stats(te$Errors)
teo = te[!te$Errors %in% boxplot.stats(te$Errors)$out,] # how to remove outliers
regin1 = regStraight(teo)
regin2 = regInt(teo)
anova(regin1, regin2, test='Chisq')
printReg(regin2, 'Regression on Number of Errors in Last Testing, Outliers Removed')
summary(regin)
jpeg('exp5testingno-outliers-bean.jpg')
beanplot(Errors ~ Stem + Tier, data = teo, main = 'Beanplot of Errors in Last
	 Testing, Outliers Removed', 
         xlab = 'Conditions (Stem [True or False]/Tier Frequency)', 
         ylab = 'Error rate (percent)', names = exp5names)
dev.off()


print(xtable(summary(regin), label = 'Regression on Error Rates in Last Testing,
	     Outliers Removed', digits = c(0, 3,3,3,3)), table.placement =
	     'tbp', caption.placement = 'top')

regnewtier = glm(Errors ~ Tier, data = te)
summary(regnewtier)


# visualizing data

beanplot(Errors ~ Stem, data = tr)
beanplot(Errors ~ Stem, data = te)
beanplot(Con ~ Stem, data = con)
beanplot(Errors ~ Tier, data = tr)
beanplot(Errors ~ Tier, data = te)
beanplot(Con ~ Tier, data = con)

