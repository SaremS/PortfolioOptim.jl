abstract type AbstractBlackLittermanViews end


struct BlackLittermanViews{T<:AbstractFloat} <:AbstractBlackLittermanViews

    μView::Vector{T}
    ΣView::Matrix{T}

    pickMatrix::Matrix{T}


    function BlackLittermanViews{T}(μView::Vector{T},ΣView::Matrix{T},pickMatrix::Matrix{T}) where T<:AbstractFloat
        len_μView = length(μView)
        @assert len_μView > 0.0
        @assert size(μView)[1] == size(ΣView)[1]
        @assert issymmetric(ΣView)
        @assert eigmin(ΣView)>0.
        new{T}(μView, ΣView, pickMatrix)
    end
end


BlackLittermanViews(μView::Vector{T}, ΣView::Matrix{T}, pickMatrix::Matrix{T}) where T<:AbstractFloat = BlackLittermanViews{T}(μView, ΣView, pickMatrix)


function BlackLittermanAbsoluteViews(μView::Vector{T}, certainty::Vector{T}) where {T<:AbstractFloat}

    μLength = n_assets = size(μView)[1]
    @assert μLength > 0
    @assert μLength == size(certainty)[1]

    μView_full = collect(skipmissing(μView))
    certainty_full = collect(skipmissing(certainty))

    μLength_full = n_views = size(μView_full)[1]

    @assert μLength_full > 0


    #####pick matrix#####
    pickMatrix = zeros((n_views, n_assets))

    a = 1

    for i in 1:n_assets
        if  !ismissing(μView[i])
            pickMatrix[a,i] = 1.0
            a += 1
        end
    end

    #####uncertainty matrix######
    ΣView = diagm(0=>certainty_full)


    return BlackLittermanViews(μView_full,ΣView, pickMatrix)

end



function calculate_BlackLittermanPosteriorPortfolio(portfolio::Portfolio, blViews::BlackLittermanViews, τ::Float64=1.)

    Σ = portfolio.Σ
    P = blViews.pickMatrix
    Ω = blViews.ΣView
    Π = portfolio.μVec
    Q = blViews.μView



    M = inv(inv(τ*Σ)+transpose(P)*inv(Ω)*P)
    numerat = inv(τ*Σ)*Π+transpose(P)*inv(Ω)*Q

    μStar = M*numerat

    posteriorDistribution = MvNormal(μStar, M)

    return Portfolio(posteriorDistribution)

end
