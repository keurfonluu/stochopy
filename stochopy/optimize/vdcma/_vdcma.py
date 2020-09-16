import numpy

from ..cmaes._cmaes import converge
from ..cmaes._constraints import _constraints_map
from .._common import messages, parallelize
from .._helpers import register, OptimizeResult

__all__ = [
    "minimize",
]


def minimize(
    fun,
    bounds,
    x0=None,
    args=(),
    maxiter=100,
    popsize=10,
    sigma=0.1,
    muperc=0.5,
    xmean0=None,
    xtol=1.0e-8,
    ftol=1.0e-8,
    constraints=None,
    workers=1,
):
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if numpy.ndim(bounds) != 2:
        raise ValueError()

    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Initial guess x0
    if x0 is not None:
        if numpy.ndim(x0) != 2 or numpy.shape(x0)[1] != ndim:
            raise ValueError()

    # Standardize and unstandardize
    xm = 0.5 * (upper + lower)
    xstd = 0.5 * (upper - lower)
    standardize = lambda x: (x - xm) / xstd
    unstandardize = lambda x: x * xstd + xm

    # CMA-ES parameters
    if sigma <= 0.0:
        raise ValueError()

    if not 0.0 < muperc <= 1.0:
        raise ValueError()

    if xmean0 is not None:
        if numpy.ndim(xmean0) != 1 or len(xmean0) != ndim:
            raise ValueError()

    # Constraints
    if constraints is not None:
        cons = _constraints_map[constraints]

    # Parallel
    funstd = parallelize(fun, args, True, workers)
    fun = lambda x: funstd(unstandardize(x))

    # Initialize arrays
    xall = numpy.empty((popsize, ndim, maxiter))
    funall = numpy.empty((popsize, maxiter))

    # Initial mean
    xmean = (
        numpy.random.uniform(-1.0, 1.0, ndim)
        if xmean0 is None
        else standardize(xmean0)
    )
    xold = numpy.empty(ndim)

    # Number of parents
    mu = int(muperc * popsize)
    
    # Strategy parameter setting: Selection
    weights = numpy.log(mu + 0.5) - numpy.log(numpy.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = weights.sum() ** 2 / numpy.square(weights).sum()
    
    # Strategy parameter setting: Adaptation
    cc = (4.0 + mueff / ndim) / (ndim + 4.0 + 2.0 * mueff / ndim)
    cfactor = (ndim - 5.0) / 6.0
    c1 = cfactor * 2.0 / ((ndim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, cfactor * 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((ndim + 2.0) ** 2 + mueff))
    
    # Initialize dynamic (internal) strategy parameters and constants
    flg_injection = False
    cs = 0.3
    ds = numpy.sqrt(ndim)
    dx = numpy.zeros(ndim)
    ps = 0.0
    dvec = numpy.ones(ndim)
    vvec = numpy.random.normal(0.0, 1.0, ndim) / numpy.sqrt(ndim)
    norm_v2 = numpy.dot(vvec, vvec)
    norm_v = numpy.sqrt(norm_v2)
    vn = vvec / norm_v
    vnn = vn ** 2
    pc = numpy.zeros(ndim)
    
    # Initialize boundaries weights
    bnd_weights = numpy.zeros(ndim)
    dfithist = numpy.ones(1)

    # Initialize arrays
    xall = numpy.empty((popsize, ndim, maxiter))
    funall = numpy.empty((popsize, maxiter))
    
    # VD-CMA
    nfev = 0
    arbestfitness = numpy.zeros(maxiter)
    ilim = int(10 + 30 * ndim / popsize)
    insigma = sigma
    validfitval = False
    iniphase = True
    
    it = 0
    converged = False
    while not converged:
        it += 1
        
        # Generate lambda offsprings
        arz = numpy.random.randn(popsize, ndim)
        ary = dvec * (arz + (numpy.sqrt(1.0 + norm_v2) - 1.0) * numpy.outer(numpy.dot(arz, vn), vn))
        if flg_injection:
            ddx = dx / dvec
            mnorm = (ddx ** 2).sum() - numpy.dot(ddx, vvec) ** 2 / (1.0 + norm_v2)
            dy = numpy.linalg.norm(numpy.random.randn(ndim)) / numpy.sqrt(mnorm) * dx
            ary[0] = dy
            ary[1] = -dy
        arx = xmean + sigma * ary
        arxvalid = arx.copy()
        diagC = numpy.diag(numpy.dot(numpy.dot(numpy.diag(dvec), numpy.eye(ndim) + numpy.outer(vvec, vvec)), numpy.diag(dvec)))
            
        # Evaluate fitness
        if constraints == "Penalize":
            arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase = cons(
                arxvalid, arx, xmean, xold, sigma, diagC, mueff, it, bnd_weights, dfithist, validfitval, iniphase, fun
            )
        else:
            arfitness = fun(arxvalid)
        nfev += popsize

        xall[:, :, it - 1] = unstandardize(arxvalid)
        funall[:, it - 1] = arfitness.copy()
        
        # Sort by fitness and compute weighted mean into xmean
        arindex = numpy.argsort(arfitness)
        dx = numpy.dot(weights, arx[arindex[:mu]]) - weights.sum() * xmean
        xold = xmean.copy()
        xmean += dx
        
        # Save best fitness
        arbestfitness[it - 1] = arfitness[arindex[0]]

        # Update sigma
        if flg_injection:
            alpha_act = numpy.where(arindex == 1)[0][0] - numpy.where(arindex == 0)[0][0]
            alpha_act /= popsize - 1.0
            ps += cs * (alpha_act - ps)
            sigma *= numpy.exp(ps / ds)
            cond = ps < 0.5
        else:
            flg_injection = True
            cond = True

        # Cumulation
        pc *= (1.0 - cc)
        pc += numpy.sqrt(cc * (2.0 - cc) * mueff) * numpy.dot(weights, ary[arindex[:mu]]) if cond else 0.0

        # Alpha and related variables
        gamma = 1.0 / numpy.sqrt(1.0 + norm_v2)
        alpha = numpy.sqrt(norm_v2 ** 2 + (1.0 + norm_v2) / vnn.max() * (2.0 - gamma)) / (2.0 + norm_v2)
        if alpha < 1.0:
            beta = (4.0 - (2.0 - gamma) / vnn.max()) / (1.0 + 2.0 / norm_v2) ** 2
        else:
            alpha = 1.
            beta = 0.
        bsca = 2.0 * alpha ** 2 - beta
        avec = 2.0 - (bsca + 2.0 * alpha ** 2) * vnn
        invavnn = vnn / avec
        
        # Rank-mu
        if cmu == 0.0:
            pvec_mu = numpy.zeros(ndim)
            qvec_mu = numpy.zeros(ndim)
        else:
            pvec_mu, qvec_mu = pvec_and_qvec(vn, norm_v2, ary[arindex[:mu]] / dvec, weights)
            
        # Rank-one
        if c1 == 0.:
            pvec_one = numpy.zeros(ndim)
            qvec_one = numpy.zeros(ndim)
        else:
            pvec_one, qvec_one = pvec_and_qvec(vn, norm_v2, pc / dvec)
            
        # Add rank-one and rank-mu before computing the natural gradient
        pvec = cmu * pvec_mu
        qvec = cmu * qvec_mu
        if cond:
            pvec += c1 * pvec_one
            qvec += c1 * qvec_one

        # Natural gradient
        if cmu + c1 > 0.0:
            ngv, ngd = ngv_ngd(dvec, vn, vnn, norm_v, norm_v2, alpha, avec, bsca, invavnn, pvec, qvec)

            # Truncation factor to guarantee at most 70 percent change
            upfactor = 1.0
            upfactor = min(upfactor, 0.7 * norm_v / numpy.sqrt(numpy.dot(ngv, ngv)))
            upfactor = min(upfactor, 0.7 * (dvec / numpy.abs(ngd)).min())
        else:
            ngv = numpy.zeros(ndim)
            ngd = numpy.zeros(ndim)
            upfactor = 1.0

        # Update parameters
        vvec += upfactor * ngv
        dvec += upfactor * ngd

        # Update the constants
        norm_v2 = numpy.dot(vvec, vvec)
        norm_v = numpy.sqrt(norm_v2)
        vn = vvec / norm_v
        vnn = vn ** 2

        # Check convergence
        status = converge(it, ndim, maxiter, xmean, xold, arbestfitness, arfitness, arindex, sigma, insigma, ilim, pc, xtol, ftol, diagC)
        converged = status is not None

    return OptimizeResult(
        x=unstandardize(arxvalid[arindex[0]]),
        success=status >= 0,
        status=status,
        message=messages[status],
        fun=arfitness[arindex[0]],
        nfev=nfev,
        nit=it,
        xall=xall[:, :, :it],
        funall=funall[:, :it],
    )


def pvec_and_qvec(vn, norm_v2, y, weights=None):
    y_vn = numpy.dot(y, vn)
    if weights is None:
        pvec = y ** 2 - norm_v2 / (1.0 + norm_v2) * (y_vn * y * vn) - 1.0
        qvec = y_vn * y - (0.5 * (y_vn ** 2 + 1.0 + norm_v2)) * vn

    else:
        pvec = numpy.dot(weights, y ** 2 - norm_v2 / (1.0 + norm_v2) * (y_vn * (y * vn).T).T - 1.0)
        qvec = numpy.dot(weights, (y_vn * y.T).T - numpy.outer(0.5 * ( y_vn ** 2 + 1.0 + norm_v2), vn))

    return pvec, qvec


def ngv_ngd(dvec, vn, vnn, norm_v, norm_v2, alpha, avec, bsca, invavnn, pvec, qvec):
    rvec = pvec - alpha / (1.0 + norm_v2) * ((2.0 + norm_v2) * qvec * vn - norm_v2 * numpy.dot(vn, qvec) * vnn)
    svec = rvec / avec - bsca * numpy.dot(rvec, invavnn) / (1.0 + bsca * numpy.dot(vnn, invavnn)) * invavnn
    ngv = qvec / norm_v - alpha / norm_v * ((2.0 + norm_v2) * (vn * svec) - numpy.dot(svec, vnn) * vn)
    ngd = dvec * svec

    return ngv, ngd


register("vdcma", minimize)
