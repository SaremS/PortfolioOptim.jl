abstract type AbstractPortfolio end

mutable struct Portfolio{T<:MultivariateDistribution} <: AbstractPortfolio

    Π::T
    μVec::Vector
    Σ::Matrix

    function Portfolio{T}(Π::T) where {T<:MultivariateDistribution}

        μVec = mean(Π)
        Σ = cov(Π)

        new{T}(Π, μVec, Σ)
    end

end

Portfolio(Π::T) where {T<:MultivariateDistribution} = Portfolio{T}(Π)



mutable struct PortfolioConstraints

    lower::Vector{Float64}
    upper::Vector{Float64}

    function PortfolioConstraints(lower::Vector{Float64}, upper::Vector{Float64})

        if length(lower)!=length(upper)
            throw(ArgumentError("Length of lower bounds must match length of upper bounds"))
        end

        for i = 1:length(lower)
            if lower[i]>upper[i]
                throw(ArgumentError("Lower bound for asset " * string(i) * " is higher than its upper bound"))
            end
        end

        lower = lower
        upper = upper

        new(lower, upper)
    end
end



mutable struct PortfolioTargetMetric

    target_metric::String
    target::Float64

    function PortfolioTargetMetric(target_metric::String, target::Float64)
        if target_metric != "portfolio return"
            throw(ArgumentError("Target metric is currently only allowed to be 'portfolio return'"))
        end

        new(target_metric, target)

    end

end


function calculate_max_sharpe_portfolio(portfolio::Portfolio, risk_free_rate::Float64, constraints::Union{PortfolioConstraints, Nothing})

    max_return = maximum(portfolio.μVec)

    inner_optimizer = BFGS()
    result = optimize(x -> sharpe_from_target_return(x, portfolio, risk_free_rate, constraints), [0.0],[max_return], ones(1)*0.1, Fminbox(inner_optimizer))

    tangent_return = result.minimizer[1]

    tm = PortfolioTargetMetric("portfolio return", tangent_return)

    tangent_weights = mean_variance_optimization(portfolio, constraints = constraints, targetMetric = tm)

    return tangent_weights


end


function calculate_tangent_portfolio_weights(portfolio::Portfolio, risk_free_rate::Float64)

    max_return = maximum(portfolio.μVec)

    inner_optimizer = BFGS()
    result = optimize(x -> sharpe_from_target_return(x, portfolio, risk_free_rate, "long only"), [0.0],[max_return], ones(1)*0.1, Fminbox(inner_optimizer))

    tangent_return = result.minimizer[1]

    tm = PortfolioTargetMetric("portfolio return", tangent_return)

    tangent_weights = mean_variance_optimization(portfolio, constraints = "long only", targetMetric = tm)

    return tangent_weights
end



function sharpe_from_target_return(target_return::Array{Float64}, portfolio::Portfolio, risk_free_rate::Float64, constraints::Union{PortfolioConstraints, Nothing, String} = "long only")

    tr = target_return[1]
    pf_target = PortfolioTargetMetric("portfolio return", tr)

    weights = mean_variance_optimization(portfolio, constraints=constraints, targetMetric = pf_target)

    return -sharpe_ratio(weights, portfolio.μVec, portfolio.Σ, risk_free_rate)
end

function sharpe_ratio(weights::Vector{Float64}, μVec::Vector{Float64}, Σ,risk_free_rate::Float64)

    pf_return = dot(weights, μVec)
    pf_risk = sqrt(weights'*Σ*weights)

    sharpe = (pf_return - risk_free_rate)/pf_risk

    return sharpe

end



function mean_variance_optimization(portfolio::Portfolio; constraints::Union{PortfolioConstraints, Nothing, String} = "long only", targetMetric::Union{PortfolioTargetMetric, Nothing} = nothing,
    optimizer::Function = mvo_osqp)

    pf = portfolio

    result = optimizer(pf.μVec, pf.Σ, constraints, targetMetric, size(pf.μVec)[1])

    return result
end



function mvo_formulate_constraint_matrices(μVec::Array{Float64}, constraints::Union{PortfolioConstraints, Nothing, String},
    targetMetric::Union{PortfolioTargetMetric, Nothing}, n::T) where T<:Int

    #minimum constraint: weights must sum to 1
    A = reshape(fill(1., n), (1,n))
    l = [1.]
    u = [1.]


    if typeof(constraints) == String
        if constraints == "long only"
            designm_long_constraints= diagm(0=>repeat([1],n))
            lower_long_constraints = repeat([0.],n)
            upper_long_constraints = repeat([1.],n)

            A = vcat(A, designm_long_constraints)
            l = vcat(l, lower_long_constraints)
            u = vcat(u, upper_long_constraints)

        end


    elseif typeof(constraints) == PortfolioConstraints
        if size(constraints.lower)[1]!=n
            throw(ArgumentError("Number of bounds does not match number of assets"))
        end

        designm_constraints= diagm(0=>repeat([1],n))


        A = vcat(A, designm_constraints)
        l = vcat(l, constraints.lower)
        u = vcat(u, constraints.upper)
    end


    if typeof(targetMetric) == PortfolioTargetMetric

        A = vcat(A, reshape(μVec, (1,n)))
        l = vcat(l, targetMetric.target)
        u = vcat(u, targetMetric.target)

    end

    return A, l, u

end




function mvo_osqp(μVec::Array{Float64}, Σ::Matrix{Float64}, constraints::Union{PortfolioConstraints, Nothing, String}, targetMetric::Union{PortfolioTargetMetric, Nothing},
    n::T) where T<:Int

    A, l, u = mvo_formulate_constraint_matrices(μVec, constraints, targetMetric, n)

    m = OSQP.Model()

    OSQP.setup!(m; P=sparse(Σ), q=-μVec, A=sparse(A), l=l, u=u, eps_abs = 1e-20, eps_rel = 1e-20)

    results = OSQP.solve!(m)

    return results.x
end
