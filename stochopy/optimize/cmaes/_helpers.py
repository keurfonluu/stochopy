import numpy


def constrain(arxvalid, arx, xmean, xold, sigma, diagC, mueff, it, bnd_weights, dfithist, validfitval, iniphase, fun):
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
        idx = numpy.logical_and(ti, numpy.abs(tx) > 3.0 * max(1.0, numpy.sqrt(ndim / mueff) ) * sigma * numpy.sqrt(diagC))
        idx = numpy.logical_and(idx, numpy.sign(tx) == numpy.sign(xmean - xold))
        bnd_weights = numpy.array([w * 1.2 ** min(1.0, mueff / 10.0 / ndim) if i else w for i, w in zip(idx, bnd_weights)])
                
    # Calculate scaling biased to unity, product is one
    bnd_scale = numpy.exp(0.9 * (numpy.log(diagC) - numpy.log(diagC).mean()))
    
    # Assigned penalized fitness
    arfitness += numpy.dot((arxvalid - arx) ** 2, bnd_weights / bnd_scale)

    return arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase


def converge(it, ndim, maxiter, xmean, xold, arbestfitness, arfitness, arindex, sigma, insigma, B, D, C, ilim, pc, xtol, ftol):
    status = None
    i = int(numpy.floor(numpy.mod(it, ndim)))

    # Stop if maximum iteration is reached
    if it >= maxiter:
        status = -1
    
    # Stop if mean position changes less than xtol
    elif numpy.linalg.norm(xold - xmean) <= xtol and arfitness[arindex[0]] < ftol:
        status = 0
        
    # Stop if fitness is less than ftol
    elif arfitness[arindex[0]] <= ftol:
        status = 1
        
    # NoEffectAxis: stop if numerical precision problem
    elif (numpy.abs(0.1 * sigma * B[:, i] * D[i]) < 1.0e-10).all():
        status = -2
        
    # NoEffectCoord: stop if too low coordinate axis deviations
    elif (0.2 * sigma * numpy.sqrt(numpy.diag(C)) < 1.0e-10).any():
        status = -3
    
    # ConditionCov: stop if the condition number exceeds 1e14
    elif D.max() > 1.0e7 * D.min():
        status = -4
    
    # EqualFunValues: stop if the range of fitness values is zero
    elif it >= ilim and arbestfitness[it - ilim : it + 1].max() - arbestfitness[it-ilim:it+1].min() < 1.0e-10:
        status = -5
            
    # TolXUp: stop if x-changes larger than 1e3 times initial sigma
    elif (sigma * numpy.sqrt(numpy.diag(C)) > 1.0e3 * insigma).any():
        status = -6
        
    # TolFun: stop if fun-changes smaller than 1e-12
    elif it > 2 and numpy.append(arfitness, arbestfitness).max() - numpy.append(arfitness, arbestfitness).min() < 1.0e-12:
        status = -7
        
    # TolX: stop if x-changes smaller than 1e-11 times initial sigma
    elif (sigma * numpy.append(numpy.abs(pc), numpy.sqrt(numpy.diag(C)).max()) < 1.0e-11 * insigma).all():
        status = -8

    return status
