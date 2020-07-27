library(animation) 
ani.options(interval = 0.1, nmax = 250) 
op = par(mar = c(3, 3, 1, 0.5)
         , mgp = c(1.5, 0.5, 0), tcl = -0.3) 
clt.ani(type = "s")

gumballs <- rep.int(1:2, 25)
gumballs <- factor(gumballs, labels = c("Red", "Blue"))
gumballs 
sample(gumballs, size = 10, replace = T) # same i time 10 items 

sum(sample(gumballs, size = 10, replace = TRUE)== "Red") # sum all the red ones in the sample

sum(sample(gumballs, size = 1000, replace = TRUE)== "Red")
# you can saqmple from any distribution

runif(n=10, min=1, max = 5) # uniform dist
rnorm(n=10, mean = 5, sd = 1) # norm dist
rbinom(n=10, size = 1, prob = .5) # binomial dist

#-----------------------------
set.seed(5)
toastAngleData <- runif(1000, 0,180) # randon unif distr numbers gererated
head(toastAngleData) # see first 6 
tail(toastAngleData) # see last 6 
mean(toastAngleData) # population mean

hist(toastAngleData) # uniform distribution

# sampling 14 elements with replacement
sample(toastAngleData, size=14, replace = T)
mean(sample(toastAngleData, size=14, replace = T)) # mean of the sample

samplingDistribution <- # distribution of the means
  replicate(10000, mean(sample(toastAngleData, size=14, replace = T)), 
            simplify = T)

hist(samplingDistribution) # 10000 mean sample from the data
abline(v=quantile(samplingDistribution, 0.025))
abline(v=quantile(samplingDistribution, 0.975))

mean(samplingDistribution)

hist(replicate(10000, mean(sample(1:1000, size = 100, replace = T))))

quantile(0:100, probs = 0.75) #75%
qnorm(0.50)

mean(replicate(10000, mean(sample(0:100, size=3))))

qnorm(0.025)
qnorm(0.975)
qnorm(0.25)