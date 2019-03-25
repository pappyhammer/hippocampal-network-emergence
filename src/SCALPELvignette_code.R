## ---- echo = FALSE-------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ------------------------------------------------------------------------
library(scalpel)
library(Matrix)

## ------------------------------------------------------------------------
#example video is provided with the R package
#automatically locate the folder that contains "Y_1.rds"
# rawDataFolder = gsub("Y_1.rds", "", system.file("extdata", "Y_1.rds", package = "scalpel"))
rawDataFolder = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/scalpel/video_to_process/"
#define the height of the example video
videoHeight = 180
#existing folder in which to save various results
#if rerunning this code yourself, change this to an existing folder on your computer
outputFolder = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/scalpel/outputs"

#run the entire SCALPEL pipeline
miniOut = scalpel(outputFolder = outputFolder, videoHeight = videoHeight, rawDataFolder = rawDataFolder, 
                  processSeparately=TRUE, fileType = "matlab")

## ---- fig.width=5, fig.height=2.7, fig.align='center'--------------------
plotResults(scalpelOutput = miniOut)

## ------------------------------------------------------------------------
#the dimensions of the dictionaries: (number of pixels) x (number of elements)
dim(miniOut$Azero)
dim(miniOut$A)
dim(miniOut$Afilter)

## ---- fig.width=2.5, fig.height=2.7, fig.align='center'------------------
plotVideoVariance(scalpelOutput = miniOut, neuronSet = "Afilter")

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#show the brightest frame for each estimated neuron
plotBrightest(scalpelOutput = miniOut, AfilterIndex = 1)
plotBrightest(scalpelOutput = miniOut, AfilterIndex = 2)
plotBrightest(scalpelOutput = miniOut, AfilterIndex = 3)

## ---- fig.width=5, fig.height=2.7, fig.align='center'--------------------
#examine the 50th member of the cluster corresponding
#to the first element of the filtered dictionary, scalpelOutput$Afilter
plotCandidateFrame(scalpelOutput = miniOut, AfilterIndex = 1, member = 50)
#examine the representative component of the cluster
plotCandidateFrame(scalpelOutput = miniOut, AfilterIndex = 1, member = 99)

## ------------------------------------------------------------------------
#summarizes Step 0 of SCALPEL
summary(miniOut, step = 0)

## ------------------------------------------------------------------------
#summarizes Step 1 of SCALPEL
summary(miniOut, step = 1)

## ------------------------------------------------------------------------
#summarizes Step 2 of SCALPEL
summary(miniOut, step = 2)

## ------------------------------------------------------------------------
#summarizes Step 3 of SCALPEL
summary(miniOut, step = 3)

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#plot the 200th frame of the pre-processed video
plotFrame(scalpelOutput = miniOut, frame = 250)

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#simplest example with default parameters
plotFrame(scalpelOutput = miniOut, frame = 100)
#using shrinkLargest argument
plotFrame(scalpelOutput = miniOut, frame = 100, shrinkLargest = TRUE)
#now plot the raw data instead of processed, and add a custom title
plotFrame(scalpelOutput = miniOut, frame = 100, videoType = "raw", title = "Raw Data for Frame 100")

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#run Step 0 using a different output folder
#to demonstrate plotting using output from scalpelStep0
step0out = scalpelStep0(outputFolder = "~/Desktop/miniDataVersion2/", videoHeight = videoHeight, rawDataFolder = rawDataFolder)
#plot the same plot as above using output from scalpelStep0
plotFrame(scalpelOutput = step0out, frame = 100)

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#simplest example with default parameters
plotThresholdedFrame(scalpelOutput = miniOut, frame = 100, threshold = miniOut$thresholdVec[1])
#change shading to purple, add a title, and use a different threshold
plotThresholdedFrame(scalpelOutput = miniOut, frame = 100, col = "purple", threshold = miniOut$thresholdVec[2], title = "2nd Threshold Value")

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#plot the raw data and add a title
plotVideoVariance(scalpelOutput = step0out, videoType = "raw", title = "Variance of Raw Data")
#we can choose whether to outline the neurons for object from scalpel() function
#when outlining neurons, also need to choose whether to plot neurons from dictionary from Step 2 (miniOut$A) or Step 3 (miniOut$Afilter)
plotVideoVariance(scalpelOutput = miniOut, neuronSet = "Afilter", neuronsToOutline = "all")
plotVideoVariance(scalpelOutput = miniOut, neuronsToOutline = "none")

## ------------------------------------------------------------------------
miniOut$thresholdVec

## ------------------------------------------------------------------------
#use a larger minimum size, smaller maximum size, and smaller maximum width/height
#use two user-selected threshold values for image segmentation
step1out = scalpelStep1(step0Output = miniOut, minSize = 40, maxSize = 200, maxWidth = 20, maxHeight = 20, thresholdVec = c(0.05, 0.07))

## ------------------------------------------------------------------------
#we can summarize the results
summary(step1out)

## ---- fig.width=5, fig.height=2.7, fig.align='center'--------------------
#simplest example with default parameters
plotCandidateFrame(scalpelOutput = miniOut, AzeroIndex = 10)
#plot raw data instead of processed, look at results from second run of Step 1
plotCandidateFrame(scalpelOutput = step1out, AzeroIndex = 50, videoType = "raw")

## ---- fig.width=5, fig.height=2.7, fig.align='center', error=TRUE--------
#examine the 100th member of the cluster corresponding 
#to the third element of the refined dictionary, scalpelOutput$A
plotCandidateFrame(scalpelOutput = miniOut, AIndex = 3, member = 100)
#if we choose 'member' to be larger than the cluster size, we'll get an error
plotCandidateFrame(scalpelOutput = miniOut, AIndex = 3, member = 300)

## ---- fig.width=5, fig.height=2.7, fig.align='center', error=TRUE--------
#now examine the 25th member of the cluster corresponding
#to the second element of the filtered dictionary, scalpelOutput$Afilter
plotCandidateFrame(scalpelOutput = miniOut, AfilterIndex = 2, member = 25)

## ------------------------------------------------------------------------
step2out = scalpelStep2(step1Output = miniOut, cutoff = 0.08)

## ------------------------------------------------------------------------
#we can summarize the results
summary(step2out)

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#the dictionary elements for cutoff=0.18
plotSpatial(scalpelOutput = miniOut, neuronSet = "A", title = paste0("Cutoff = ", miniOut$cutoff))
#the dictionary elements for cutoff=0.08
plotSpatial(scalpelOutput = step2out, neuronSet = "A", title = paste0("Cutoff = ", step2out$cutoff))

## ---- fig.width=7.5, fig.height=2.7, fig.align='center'------------------
plotCluster(scalpelOutput = miniOut, AIndex = 1)

## ------------------------------------------------------------------------
#require elements to have at least 50 members in their clusters
step3largeCluster = scalpelStep3(step2Output = miniOut, minClusterSize = 50)

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#plot the dictionary from Step 2, A
plotSpatial(scalpelOutput = step3largeCluster, neuronSet = "A", title = "A")
#plot the filtered dictionary from Step 3, Afilter
plotSpatial(scalpelOutput = step3largeCluster, neuronSet = "Afilter", title = "Afilter")

## ------------------------------------------------------------------------
#we exclude the 4th and 5th dictionary elements, which from our plot above didn't seem to be distinct neurons
#and we choose lambda using the distribution of the processed data
step3out = scalpelStep3(step2Output = step2out, lambdaMethod = "distn", excludeReps = c(4, 5))

## ------------------------------------------------------------------------
#we can summarize the results
summary(step3out)

## ------------------------------------------------------------------------
#alternatively, we can directly specify lambda
step3out_pickLambda = scalpelStep3(step2Output = step2out, lambda = 0.05, excludeReps = c(4, 5))

## ---- fig.width=5, fig.height=2.7, fig.align='center'--------------------
#plot the results of only neurons 1 and 3, with no numbering
#use custom colors, custom titles, and only draw the outlines of neurons
plotResults(scalpelOutput = step3out, neuronsToDisplay = c(1, 3), colVec = c("red", "blue", "orange"), number = FALSE, border = TRUE, titleA = "Neurons 1 & 3", titleZ = "Calcium over time", ylabZ = "Neuron")

## ---- fig.width=2.5, fig.height=2.7, fig.align='center'------------------
#plot a zoomed-in view of one of the preliminary dictionary elements
plotSpatial(A = miniOut$Azero, videoHeight = videoHeight, zoom = TRUE, neuronsToDisplay = 1)

## ---- fig.show='hold', fig.width=2.5, fig.height=2.7, fig.align='center'----
#non-default values for all arguments:
#plot the third brightest frame ('brightIndex=3') for neuron 2
#using the raw data and only outline that neuron ('neuronsToOutline="main"')
#also use a custom title
plotBrightest(scalpelOutput = miniOut, AfilterIndex = 2, videoType = "raw", neuronsToOutline = "main", brightIndex = 3, title = "Neuron 2")

## ------------------------------------------------------------------------
miniOut$version

## ------------------------------------------------------------------------
#retrieve previous results
miniOutCopy = getScalpel(outputFolder = outputFolder, version = miniOut$version)

## ------------------------------------------------------------------------
identical(miniOut, miniOutCopy)

## ------------------------------------------------------------------------
#retrieve results for cutoff=0.08, lambdaMethod="distn", and excluding components 4 and 5
smallerCutoffOut = getScalpel(outputFolder = outputFolder, version = miniOut$version, cutoff = 0.08, lambdaMethod = "distn", excludeReps = c(4, 5))

## ------------------------------------------------------------------------
step1outCopy = getScalpelStep1(outputFolder = outputFolder, version = step1out$version)
#check that the results are identical
identical(step1out, step1outCopy)

## ---- eval=FALSE---------------------------------------------------------
## ###NOT RUN
## ###THESE COMMANDS WERE NOT RUN AS PART OF THIS VIGNETTE
## ###THE FOLLOWING COMMANDS ARE INTERACTIVE
## ###SEE THE VIDEO AT ajpete.com/software FOR HOW TO USE THESE COMMANDS
## #do an initial review of each estimated neuron
## reviewNeuronsInteractive(scalpelOutput = miniOut, neuronSet = "A")
## #gather more information as to whether an estimated neuron is real
## reviewNeuronsMoreFrames(scalpelOutput = miniOut, neuronSet = "A")
## reviewOverlappingNeurons(scalpelOutput = miniOut, neuronSet = "A")
## #update the status of the neurons we were unsure about
## updateNeuronsInteractive(scalpelOutput = miniOut, neuronSet = "A")

## ---- eval=FALSE---------------------------------------------------------
## ###NOT RUN
## ###THIS COMMAND WAS NOT RUN AS PART OF THIS VIGNETTE
## ###SEE THE VIDEO AT ajpete.com/software FOR HOW TO USE THIS COMMAND
## step3discarded = scalpelStep3(step2Output = miniOut, excludeReps = "discarded")

## ---- eval=FALSE---------------------------------------------------------
## ###NOT RUN
## ###THESE COMMANDS WERE NOT RUN AS PART OF THIS VIGNETTE
## ###THE FOLLOWING COMMANDS ARE INTERACTIVE
## ###SEE THE VIDEO AT ajpete.com/software FOR HOW TO USE THESE COMMANDS
## #save the plots for determining the classifications
## reviewNeurons(scalpelOutput = scalpelOutput, neuronSet = "A")
## #update the status of the neurons once files are sorted
## updateNeurons(scalpelOutput = scalpelOutput, neuronSet = "A")
## #gather more information as to whether an estimated neuron is real
## reviewNeuronsMoreFrames(scalpelOutput = scalpelOutput, neuronSet = "A")
## reviewOverlappingNeurons(scalpelOutput = scalpelOutput, neuronSet = "A")
## #update the status of the neurons once files are sorted again
## updateNeurons(scalpelOutput = scalpelOutput, neuronSet = "A")

## ---- eval=FALSE---------------------------------------------------------
## ###NOT RUN
## ###THESE COMMANDS WERE NOT RUN AS PART OF THIS VIGNETTE
## ###SEE THE VIDEO AT ajpete.com/software FOR HOW TO USE THESE COMMANDS
## #dictionary prior to manual classification
## plotSpatial(scalpelOutput = miniOut, neuronSet = "A")
## #dictionary after to manual classification
## plotSpatial(scalpelOutput = miniOut, neuronSet = "A", neuronsToDisplay = "kept")

## ---- eval=FALSE---------------------------------------------------------
## ###NOT RUN
## ###THIS COMMAND WAS NOT RUN AS PART OF THIS VIGNETTE
## ###SEE THE VIDEO AT ajpete.com/software FOR HOW TO USE THIS COMMAND
## getNeuronStatus(scalpelOutput = miniOut, neuronSet = "A")

## ---- eval=FALSE---------------------------------------------------------
## ###NOT RUN
## ###THESE COMMANDS WERE NOT RUN AS PART OF THIS VIGNETTE
## ###SEE THE VIDEO AT ajpete.com/software FOR HOW TO USE THESE COMMANDS
## #plotting functions that use the neuronsToOutline="kept" argument
## plotVideoVariance(scalpelOutput = miniOut, neuronSet = "Afilter", neuronsToOutline = "kept")
## plotBrightest(scalpelOutput = miniOut, AfilterIndex = 1, neuronsToOutline = "kept")
## #plotting functions that use the neuronsToDisplay="kept" argument
## plotResults(scalpelOutput = miniOut, neuronsToDisplay = "kept")
## plotResultsAllLambda(scalpelOutput = miniOut, neuronsToDisplay = "kept")
## plotSpatial(scalpelOutput = miniOut, neuronSet = "Afilter", neuronsToDisplay = "kept")
## plotTemporal(scalpelOutput = miniOut, neuronsToDisplay = "kept")

## ------------------------------------------------------------------------
processedY = getY(scalpelOutput = miniOut)

## ------------------------------------------------------------------------
rawY = getY(scalpelOutput = miniOut, videoType = "raw")

## ---- fig.width=2.5, fig.height=2.7, fig.align='center'------------------
#video data is automatically read in
plotFrame(scalpelOutput = miniOut, frame = 100, videoType = "processed")

## ---- fig.width=2.5, fig.height=2.7, fig.align='center'------------------
#video data is provided as 'Y' argument
plotFrame(scalpelOutput = miniOut, frame = 100, Y = processedY)

## ---- out.width="600px", echo=FALSE--------------------------------------
knitr::include_graphics("Rpackage_flowchart_v2.jpg")

## ---- out.width="400px", echo=FALSE--------------------------------------
knitr::include_graphics("Rpackage_table_v2.jpg")

