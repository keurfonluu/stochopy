import numpy


def Penalize(arxvalid, arx, xmean, xold, sigma, diagC, mueff, it, bnd_weights, dfithist, validfitval, iniphase, fun):
    popsize, ndim = arx.shape
    ones = numpy.ones(ndim)

    # Clip to boundaries
    arxvalid = numpy.where(arxvalid < -1.0, -ones, arxvalid)
    arxvalid = numpy.where(arxvalid > 1.0, ones, arxvalid)
    arfitness = fun(arxvalid)
    
    # Get delta fitness values
    perc = numpy.percentile(arfitness, [25.0, 75.0])
    delta = (perc[1] - perc[0]) / ndim / diagC.mean() / sigma ** 2
    
    # Catch non-sensible values
    if delta == 0:
        delta = dfithist[dfithist > 0.0].min()
    elif not validfitval:
        dfithist = numpy.empty(0)
        validfitval = True
        
    # Store delta fitness values
    if dfithist.size < 20 + (3.0 * ndim) / popsize:
        dfithist = numpy.append(dfithist, delta)
    else:
        dfithist = numpy.append(dfithist[1 : dfithist.size + 1], delta)
        
    # Corrected mean
    ti = numpy.logical_or(xmean < -ones, xmean > ones)
    tx = numpy.where(xmean < -ones, -ones, xmean)
    tx = numpy.where(xmean > ones, ones, xmean)
    
    # Set initial weights
    if iniphase and ti.any():
        bnd_weights.fill(2.0002 * numpy.median(dfithist))
        if validfitval and it > 2:
            iniphase = False
                
    if ti.any():
        tx = xmean - tx
        idx = numpy.logical_and(ti, numpy.abs(tx) > 3.0 * max(1.0, numpy.sqrt(ndim / mueff)) * sigma * numpy.sqrt(diagC))
        idx = numpy.logical_and(idx, numpy.sign(tx) == numpy.sign(xmean - xold))
        bnd_weights = numpy.array([w * 1.2 ** min(1.0, mueff / 10.0 / ndim) if i else w for i, w in zip(idx, bnd_weights)])
                
    # Calculate scaling biased to unity, product is one
    bnd_scale = numpy.exp(0.9 * (numpy.log(diagC) - numpy.log(diagC).mean()))
    
    # Assigned penalized fitness
    arfitness += numpy.dot((arxvalid - arx) ** 2, bnd_weights / bnd_scale)

    return arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase


_constraints_map = {
    "Penalize": Penalize,
}
