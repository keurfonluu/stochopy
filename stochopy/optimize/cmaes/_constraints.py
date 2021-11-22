import numpy as np


def Penalize(
    arxvalid,
    arx,
    xmean,
    xold,
    sigma,
    diagC,
    mueff,
    it,
    bnd_weights,
    dfithist,
    validfitval,
    iniphase,
    fun,
):
    """
    Penalize function values of infeasible solutions.

    Box constraint handling by adding a penalty term that quantifies the distance of the
    parameters from the feasible space.

    """
    popsize, ndim = arx.shape
    ones = np.ones(ndim)

    # Clip to boundaries
    arxvalid = np.where(arxvalid < -1.0, -ones, arxvalid)
    arxvalid = np.where(arxvalid > 1.0, ones, arxvalid)
    arfitness = fun(arxvalid)

    # Get delta fitness values
    perc = np.percentile(arfitness, [25.0, 75.0])
    delta = (perc[1] - perc[0]) / ndim / diagC.mean() / sigma ** 2

    # Catch non-sensible values
    if delta == 0:
        delta = dfithist[dfithist > 0.0].min()
    elif not validfitval:
        dfithist = np.empty(0)
        validfitval = True

    # Store delta fitness values
    if dfithist.size < 20 + (3.0 * ndim) / popsize:
        dfithist = np.append(dfithist, delta)
    else:
        dfithist = np.append(dfithist[1 : dfithist.size + 1], delta)

    # Corrected mean
    ti = np.logical_or(xmean < -ones, xmean > ones)
    tx = np.where(xmean < -ones, -ones, xmean)
    tx = np.where(xmean > ones, ones, xmean)

    # Set initial weights
    if iniphase and ti.any():
        bnd_weights.fill(2.0002 * np.median(dfithist))
        if validfitval and it > 2:
            iniphase = False

    if ti.any():
        tx = xmean - tx
        idx = np.logical_and(
            ti,
            np.abs(tx) > 3.0 * max(1.0, np.sqrt(ndim / mueff)) * sigma * np.sqrt(diagC),
        )
        idx = np.logical_and(idx, np.sign(tx) == np.sign(xmean - xold))
        bnd_weights = np.array(
            [
                w * 1.2 ** min(1.0, mueff / 10.0 / ndim) if i else w
                for i, w in zip(idx, bnd_weights)
            ]
        )

    # Calculate scaling biased to unity, product is one
    bnd_scale = np.exp(0.9 * (np.log(diagC) - np.log(diagC).mean()))

    # Assigned penalized fitness
    arfitness += np.dot((arxvalid - arx) ** 2, bnd_weights / bnd_scale)

    return arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase


_constraints_map = {
    "Penalize": Penalize,
}
