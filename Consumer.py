
#creating the consumer class


from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ConsumerClass:
    """ a consumer with nested CES preferences over three goods

    Good 1 is food. Goods 2 and 3 are bus trips and train trips, and they sit
    together in a nest.

    The problem is written in *nested* budget shares, in the same two steps as
    the nests themselves:

        s1 = the share of income spent on food
        w  = the share of the remaining (travel) budget spent on the bus

    so that s2 = (1-s1)*w and s3 = (1-s1)*(1-w). Any (s1,w) in the unit square
    is a possible choice, and every possible choice is in the unit square, so
    the constraint set is exactly the box that L-BFGS-B takes as `bounds`.
    """

    def __init__(self,par=None):

        # a. setup
        self.setup()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        """ set the baseline parameters """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preference weights
        par.alpha = 0.60 # weight on food
        par.beta = 0.50 # weight on the bus

        # b. substitution
        par.sigma_A = 0.80 # between food and travel (upper nest)
        par.sigma_B = 0.40 # between bus and train (lower nest)

        # c. prices and income
        par.p1 = 1.0 # price of food
        par.p2 = 1.0 # price of a bus trip
        par.p3 = 1.5 # price of a train trip
        par.I = 10.0 # income

        # d. numerical settings
        par.s_min = 1e-12 # smallest quantity allowed in .ces(), see the note there

    def __str__(self):
        """ print the parameters """

        par = self.par

        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.4f}, beta = {par.beta:.4f}')
        lines.append(f'  sigma_A = {par.sigma_A:.4f}, sigma_B = {par.sigma_B:.4f}')
        lines.append(f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, p3 = {par.p3:.4f}')
        lines.append(f'  I = {par.I:.4f}')
        return '\n'.join(lines)

    ###################
    # 1. the CES nest #
    ###################

    def ces(self,z1,z2,w,sigma):
        """ the CES aggregate of two inputs

        Computes (w*z1**rho + (1-w)*z2**rho)**(1/rho) with rho = 1-1/sigma.

        The inputs are floored at par.s_min, because z**rho is not defined for
        z = 0 when rho < 0, and the corners of the unit square do give zeros.

        Note on the size of par.s_min. It must be *far* below the step that
        L-BFGS-B uses to estimate the gradient, which is `eps` and about 1e-8 by
        default. Otherwise both z and z+eps get floored to the same number near a
        bound, utility comes out exactly equal in the two points, the estimated
        gradient is zero, and the solver stops on the bound and stays there. With
        s_min = 1e-8 that really happens: the answers in section 4 for high tax
        rates come out as zero revenue. 1e-12 is small enough to be invisible and
        large enough to keep z**rho from overflowing.

        Args:
            z1 (float or ndarray): first input
            z2 (float or ndarray): second input
            w (float): weight on the first input
            sigma (float): substitution parameter, must not be 1

        Returns:
            (float or ndarray): the CES aggregate
        """

        par = self.par

        assert not np.isclose(sigma,1.0), 'sigma = 1 gives rho = 0 and a division by zero'

        z1 = np.maximum(z1,par.s_min)
        z2 = np.maximum(z2,par.s_min)

        rho = 1-1/sigma
        return (w*z1**rho + (1-w)*z2**rho)**(1/rho)

    def utility(self,x1,x2,x3):
        """ nested CES utility of a bundle of quantities

        Two steps: first combine goods 2 and 3 into the travel composite, then
        combine good 1 and the composite into utility. Use .ces() for both.

        Args:
            x1 (float or ndarray): quantity of good 1
            x2 (float or ndarray): quantity of good 2
            x3 (float or ndarray): quantity of good 3

        Returns:
            (float or ndarray): utility
        """

        par = self.par

        # a. the travel composite: bus and train combined (eq. 1)
        xB = self.ces(x2,x3,par.beta,par.sigma_B)

        # b. utility: food and travel combined (eq. 2)
        u = self.ces(x1,xB,par.alpha,par.sigma_A)

        return u

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self,s1,w):
        """ the three budget shares implied by the nested shares

        Args:
            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:
            (tuple): the three budget shares, which always sum to one
        """

        # s1 is the food share directly; the remaining (1-s1) travel budget is
        # split between bus (share w) and train (share 1-w), so the three
        # shares add up to s1 + (1-s1)*w + (1-s1)*(1-w) = 1 for any (s1,w)
        return s1,(1-s1)*w,(1-s1)*(1-w)

    def quantities(self,s1,w):
        """ the quantities implied by the nested shares

        Args:
            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:
            (tuple): the three quantities
        """

        par = self.par

        # a. get the three budget shares
        s1,s2,s3 = self.shares(s1,w)

        # b. convert each share of income into a quantity: x_j = s_j*I/p_j
        return s1*par.I/par.p1, s2*par.I/par.p2, s3*par.I/par.p3

    def value_of_choice(self,s1,w):
        """ utility of the bundle implied by the nested shares

        Args:
            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:
            (float or ndarray): utility
        """

        # a. the quantities the consumer buys, given the nested shares
        x1,x2,x3 = self.quantities(s1,w)

        # b. utility of that bundle
        u = self.utility(x1,x2,x3)

        return u

    def objective(self,s):
        """ minus utility, for a minimizer

        Nothing else is needed: the bounds are the whole constraint.

        Args:
            s (ndarray): array of length 2 with (s1,w)

        Returns:
            (float): minus utility
        """

        # scipy.optimize.minimize always *minimizes*, so to *maximize* utility
        # we hand it minus utility instead -- minimizing -u is the same as
        # maximizing u
        return -self.value_of_choice(s[0],s[1])

    #################
    # 3. solving it #
    #################

    def solve_grid(self,N=200,do_print=True):
        """ solve by a 2-dimensional grid search over the nested shares

        Every point of the unit square is a possible choice, so there is nothing
        to mask out here -- a plain np.argmax will do.

        Args:
            N (int): number of grid points for each variable
            do_print (bool): print the solution

        Returns:
            (SimpleNamespace): the grids, the utility values and the best point
        """

        par = self.par
        opt = SimpleNamespace()

        # a. the two grids
        s1_vec = np.linspace(0,1,N)
        w_vec = np.linspace(0,1,N)
        s1_grid,w_grid = np.meshgrid(s1_vec,w_vec,indexing='ij')

        # b. utility in every grid point
        u_grid = self.value_of_choice(s1_grid,w_grid)

        # c. the best point
        i_max = np.argmax(u_grid)
        i1,i2 = np.unravel_index(i_max,u_grid.shape)

        # d. results
        opt.N = N
        opt.evaluations = N*N
        opt.s1 = s1_grid[i1,i2]
        opt.w = w_grid[i1,i2]
        opt.s1,opt.s2,opt.s3 = self.shares(opt.s1,opt.w)
        opt.u = u_grid[i1,i2]
        opt.s1_grid = s1_grid
        opt.w_grid = w_grid
        opt.u_grid = u_grid

        if do_print:
            print(f'grid search (N={N}): s1 = {opt.s1:.6f}, w = {opt.w:.6f}, u = {opt.u:.6f}')
            print(f'  shares: s1 = {opt.s1:.4f}, s2 = {opt.s2:.4f}, s3 = {opt.s3:.4f} (sum = {opt.s1+opt.s2+opt.s3:.4f})')

        return opt

    def solve(self,s0=None,do_print=True,**kwargs):
        """ solve with L-BFGS-B

        The bounds are ((0,1),(0,1)) -- the whole constraint set.

        Args:
            s0 (ndarray): starting guess for (s1,w)
            do_print (bool): print the solution
            kwargs: passed on to optimize.minimize, e.g. options={'ftol':1e-10}

        Returns:
            (SimpleNamespace): the solution and the convergence path
        """

        par = self.par
        opt = SimpleNamespace()

        # a. starting guess
        if s0 is None: s0 = np.array([0.5,0.5])
        s0 = np.asarray(s0,dtype=float)

        # b. record the path with a callback
        path = [s0.copy()]
        def callback(sk):
            path.append(sk.copy())

        # c. minimize
        res = optimize.minimize(self.objective,s0,method='L-BFGS-B',
                                 bounds=((0,1),(0,1)),callback=callback,**kwargs)

        # d. results
        opt.s1,opt.w = res.x
        opt.s1,opt.s2,opt.s3 = self.shares(opt.s1,opt.w)
        opt.u = -res.fun
        opt.path = np.array(path)
        opt.res = res

        if do_print:
            print(f'L-BFGS-B: s1 = {opt.s1:.6f}, w = {opt.w:.6f}, u = {opt.u:.6f}, '
                  f'nfev = {res.nfev}, nit = {res.nit}, success = {res.success}')

        return opt

#creating the government class

from types import SimpleNamespace
import numpy as np
from scipy import optimize

from Consumer import ConsumerClass

class GovernmentClass(ConsumerClass):
    """ a government raising revenue from the consumer in Consumer.py

    Two kinds of instrument:
    1) a lump-sum tax T, which reduces income
    2) product taxes tau1, tau2, tau3, which raise the prices

    The consumer is the ConsumerClass, so everything from there is inherited.

    NOTE: this file belongs to section 4 of the project (Lump-sum taxes and
    product taxes) and is not needed for section 2 (Solving the model
    numerically). It is kept here, untouched from the starting-point version,
    ready to be filled in when that section is done.
    """

    def __init__(self,par=None):

        # a. default setup
        self.setup()
        self.setup_government()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. remember the situation without taxes
        #    this must happen *after* b., or a changed price would not be picked up
        self.sync_pre_tax()

    def setup_government(self):
        """ add the tax instruments to the parameters """

        par = self.par

        # a. lump-sum tax
        par.T = 0.0 # lump-sum tax (a transfer if negative)

        # b. product taxes
        par.tau1 = 0.0 # tax rate on food
        par.tau2 = 0.0 # tax rate on bus trips
        par.tau3 = 0.0 # tax rate on train trips

    def sync_pre_tax(self):
        """ store the current prices and income as the situation without taxes

        Revenue is collected at these prices, so they must be the ones *before*
        any taxes were added.
        """

        par = self.par

        par.p1_pre = par.p1
        par.p2_pre = par.p2
        par.p3_pre = par.p3
        par.I_pre = par.I

    ##############################
    # 1. what the consumer faces #
    ##############################

    def set_taxes(self,T=0.0,tau1=0.0,tau2=0.0,tau3=0.0):
        """ set the taxes, and update the prices and the income the consumer faces

        The price the consumer pays for good j is (1+tau_j) times the price the
        seller receives, and income is reduced by the lump-sum tax. After this
        call every method inherited from ConsumerClass -- .solve(), .shares(),
        .value_of_choice(), .solve_grid() -- automatically refers to the
        situation *with* taxes.

        Args:
            T (float): lump-sum tax
            tau1 (float): tax rate on food
            tau2 (float): tax rate on bus trips
            tau3 (float): tax rate on train trips
        """

        par = self.par

        # a. remember the taxes
        par.T = T
        par.tau1 = tau1
        par.tau2 = tau2
        par.tau3 = tau3

        # b. the prices the consumer pays
        par.p1 = (1+tau1)*par.p1_pre
        par.p2 = (1+tau2)*par.p2_pre
        par.p3 = (1+tau3)*par.p3_pre

        # c. income after the lump-sum tax
        par.I = par.I_pre - T

    #########################################
    # 2. revenue, and what the consumer gets #
    #########################################

    def tax_revenue(self,opt=None):
        """ total tax revenue given the taxes currently set

        Note that revenue is collected at the prices the *seller* receives, so
        the tax paid on good j is tau_j*p_j_pre*x_j.

        Args:
            opt (SimpleNamespace): a solution from .solve(). Solved for here if
                not given -- pass it in when you already have it, to avoid
                solving the same problem twice

        Returns:
            (float): tax revenue
        """

        par = self.par

        # a. what does the consumer buy, given the taxes?
        #    .quantities() takes the nested shares (s1,w) from the solution
        if opt is None: opt = self.solve(do_print=False)
        pass

        # b. the lump-sum tax, plus the product tax on each good
        return R

    def revenue_and_utility(self,tau,goods=(2,)):
        """ revenue and utility when the same tax rate is put on each good in goods

        Args:
            tau (float): the common tax rate
            goods (tuple): which goods to tax, e.g. (2,) or (2,3) or (1,2,3)

        Returns:
            (tuple): (revenue, utility)
        """

        pass
        return R,u

    def revenue_and_utility_lump_sum(self,T):
        """ the same, for a lump-sum tax of T

        Args:
            T (float): the lump-sum tax

        Returns:
            (tuple): (revenue, utility)
        """

        pass
        return R,u

    ##########################################
    # 3. hitting a given revenue requirement #
    ##########################################

    def max_revenue(self,goods=(2,),tau_max=10.0,N=1001):
        """ the largest revenue this instrument can ever raise

        A grid over the tax rate is enough, exactly as in section 2.1: compute
        the revenue in every grid point and keep the best one.

        If the answer comes back at tau_max, the curve was still rising when the
        grid ran out -- there is no top in the range searched.

        Args:
            goods (tuple): which goods to tax
            tau_max (float): largest tax rate to consider
            N (int): number of grid points

        Returns:
            (tuple): (the revenue-maximizing rate, the largest revenue)
        """

        pass
        return tau,R

    def find_tax_rate(self,R_target,goods=(2,),bracket=(1e-10,1.0)):
        """ the tax rate on goods that raises exactly R_target

        Careful: revenue is not always increasing in the tax rate. There can be
        two rates that raise the same revenue, and a revenue target above the
        largest possible revenue cannot be reached at all. In that case there is
        no sign change in the bracket, and the root-finder will raise a
        ValueError -- which is the correct answer, not a bug. Catch it and
        return np.nan.

        Args:
            R_target (float): the revenue requirement
            goods (tuple): which goods to tax
            bracket (tuple): interval of tax rates to search in

        Returns:
            (float): the tax rate, or np.nan if the target cannot be reached
        """

        pass
        return tau

