# Given Min , Max and Data, Need to rescale Data's own min-max range to the given Min Max range

def linearScale(givenMin, givenMax, data):
    dataMin = min(data)
    dataMax = max(data)
    scalingFactor = (givenMax - givenMin)/(dataMax - dataMin)
    scaledData = [ scalingFactor*(x - dataMin) + givenMin for x in data]
    return scaledData

dt = range(10)
sdt = linearScale(5,10,dt)
print(sdt)
