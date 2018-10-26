using LinearAlgebra
using Distributions
using OSQP
using SparseArrays
using Optim


include("portfolio.jl")
include("blacklitterman.jl")



cova = [0.0009 0.0007 0.0009 0.0007;
        0.0007 0.0181 0.0010 0.0071;
        0.0009 0.0010 0.0015 0.0010;
        0.0007 0.0071 0.0010 0.0097]

mu = [0.05, 0.12, 0.03, 0.11]

dist = MvNormal(mu, cova)

test = Portfolio(dist)

target = PortfolioTargetMetric("portfolio return", 0.04)

constr = PortfolioConstraints([0.05,0.5,0.05,0.05],[0.9,1.0,0.9,0.9])

reshape(test.μVec,(1,4))

ax = calculate_tangent_portfolio_weights(test, 0.01)

ax2 = calculate_max_sharpe_portfolio(test, 0.01, constr)

ax2'*mu
sqrt(ax2'*cova*ax2)

result.minimizers


sharpe_ratio(ttt, test, 0.01)




dot(ttt,mu)
