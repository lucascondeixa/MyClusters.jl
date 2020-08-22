
###### NOTE: These packages might be needed, uncomment them accordingly
###### to the use:

# Pkg.add("Gurobi")
# Pkg.add("StatPlots")
# Pkg.add("JuMP")
# Pkg.add("GR")
# Pkg.add("Distances")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("StatsBase")

###### NOTE: parallelization
# Threads.nthreads()
# addprocs()

###### NOTE: Sometimes "Compat" package hampers the other packages to be
###### correctly installed
# Pkg.rm("Compat")

###### NOTE: Checking out if the packages are in place
# Pkg.build()
# Pkg.status()
# Pkg.update()

###### NOTE: Standard procedure
# using Clustering
# using GR
# using IndexedTables
# using StatPlots
using Distributed
@everywhere using Suppressor
@everywhere using SharedArrays
@everywhere using StatsBase
@everywhere using Statistics
@everywhere using Gurobi
@everywhere using CPLEX
@everywhere using JuMP
@everywhere using DataFrames
@everywhere using Distances
@everywhere using CSV
@everywhere using Random

### Kmeans algorithm
function kmeans3D(seed::Bool,
    k::Int64,
    weight::Int64,
    a₁::Vector{Float64},
    a₂::Vector{Float64},
    a₃::Vector{Float64})

    ## Clustering modelling brought by the article named "A New Approach to
    ## Model Load Levels in Electric Power Systems With High Renewable
    ## Penetration" by Sonja Wogrin et al. (2014)

    # x: Aux Array for concatenating the DataFrame used
    # Data: DataFrame used
    # k: number of classes
    # Weight: weight used to account clusters' distances
    ### Weight = 1: dif_mean
    ### Weight = 2: dif_min
    ### Weight = 3: dif_max
    # n: Number of points
    # m: Number of dimensions
    # dist: distances from k-centers (n,k)
    # class: Array of classes (n)
    # weights_ar: Dataframe of costs
    # first: Array of first indexes
    # k_cent: Array of first centroids
    # costs: Array of weighted distances
    # total_cost: Auxiliar var.
    # total_cost_new: Auxiliar var.
    # δ: Auxiliar paramenter

    ## Initialization

    n = length(a₁)                                  # Number of points
    m = 3                                           # Number of dimensions
    Data = [a₁ a₂ a₃]                               # Completing the Data with costs
    weights_ar = [zeros(n) zeros(n) zeros(n)]       # Costs array
    δ = 1e-10                                       # Aux. paramenter
    tol = 1e10                                      # Tolerance
    class_ret = Array{Float64}(undef,n,n)           # Output (1)
    k_cent_ret = Array{Float64}(undef,n,n,3)        # Output (2)
    times_ret = Array{Float64}(undef,n)             # Output (3)
    # N_transit_ret = Array{Float64}(undef,n,n,n)   # Output (4)
    first_s_ret = Array{Float64}(undef,n,n)         # Output (5)

    # First cost settings (Only for 2D)
    dif_mean = mean(Data[:,1]-Data[:,2]-Data[:,3])
    dif_min = minimum(Data[:,1]-Data[:,2]-Data[:,3])
    dif_max = maximum(Data[:,1]-Data[:,2]-Data[:,3])

    # Defining weights
    for i in 1:n
        weights_ar[i,1] = abs(Data[i,1]-Data[i,2]-Data[i,3]-dif_mean+δ)
        weights_ar[i,2] = abs(Data[i,1]-Data[i,2]-Data[i,3]-dif_min+δ)
        weights_ar[i,3] = abs(Data[i,1]-Data[i,2]-Data[i,3]-dif_max+δ)
    end

    for l in 1:k
        # texec = now()
        # println("l is: $l / $texec")

        tini = time()

        ## Initial definitions
        dist = zeros(n,l)                           # Array of distances (n,k)
        costs = zeros(n,l)                          # Array of costs (n,k)
        if seed
            Random.seed!(10^6)                      # Set the seed in the case of comparison
        end
        first_s = sample(1:n,l,replace=false)       # Sampling the first centroids
        k_cent = [a₁ a₂ a₃][first_s,:]              # First centroids
        class = zeros(Int64, n)                     # Array of classes (n)
        c_use = zeros(Bool,l)                       # Array with the using status
        total_cost = 0                              # Starting the auxiliar var.
        total_cost_new = 1                          # Starting the auxiliar var.
        cont = 0                                    # Counter

        #Assigning classes (non_parallel)
        while total_cost != total_cost_new
            cont = cont + 1
            if cont > tol
                println("\n Does not converge in l = $l \n")
            end

            total_cost = total_cost_new

            # Defining distances
            for i in 1:n
                for j in 1:l
                    dist[i,j] = evaluate(Euclidean(),
                    Data[i,1:m],k_cent[j,:])
                end
            end

            # Defining costs
            for i in 1:n
                for j in 1:l
                    costs[i,j] = weights_ar[i,weight]*dist[i,j]
                end
            end

            # Defining classes
            for i in 1:n
                class[i] = findmin(costs[i,:])[2]
            end

            ## Classification used / unused clusters
            for i in 1:l
                if length(class[class[:] .== i]) == 0
                    c_use[i] = false
                else
                    c_use[i] = true
                end
            end

            # Update classes (mean for those used and 0 for the unused)
            for j in 1:l
                if c_use[j]
                    k_cent[j,1] = mean(a₁[class[:] .== j])
                    k_cent[j,2] = mean(a₂[class[:] .== j])
                else
                    k_cent[j,1] = mean(a₁)
                    k_cent[j,2] = mean(a₂)
                end
            end

            total_cost_new = sum(costs[i,class[i]] for i in 1:n)
        end

        Size_Cluster = zeros(k)

        for i in 1:l
            if c_use[i]
                Size_Cluster[i] = length(Data[class[:] .== i,:][:,1])
            else
                Size_Cluster[i] = 0
            end
        end

        # #-------------------------------------------------------------------------------
        # ### Transition Matrix
        # #-------------------------------------------------------------------------------
        #
        # N_transit = zeros(n,n)
        # from = 0
        # to = 0
        # for i in 1:n-1
        # 	from = class[i]
        # 	to = class[i+1]
        # 	N_transit[from,to] = N_transit[from,to] + 1
        # end

        Δt = time() - tini
        times_ret[l] = Δt


        # N_transit_ret[l,:,:] = N_transit
        class_ret[l,:] = class
        k_cent_ret[l,1:l,:] = k_cent
        first_s_ret[l,1:l] = first_s

    end

    class_ret[n,:] = collect(1:n)

    return class_ret, times_ret, first_s_ret

end

### Hierarchical Clustering with both Wasserstein and Euclidean Distances as the
# indifference metric
function HCnD(k::Int64,
    dm::Int64,
	a::Array{Float64}...)

    ## dm: discrepancy metric (1: WD / 2: ED)

    for i in 1:length(a)
        if size(a[1]) != size(a[i])
            println("Different sizes input")
            return
        end
    end

	### Initialization
	n = length(a[1][:,1])                               # Total number of hours
    m = length(a[1][1,:])                               # Number of nodes
    d = length(a)                                       # Number of dimensions (attributes, e.g. DWS = 3)
	class = collect(1:n)                                # Array of classes (n)
    k_cent = zeros(n,m,d)                               # Centroids array (hours, nodes, dimensions - DWS)
	dist = zeros(n)                                     # Array of distances (n)
	l = length(a[1][:,1])                               # Number of clusters used in each iteration
	counter = 0                                         # Control the number of iterations
    class_ret = Array{Float64}(undef,n,n)               # Output (1): Classes
    k_cent_ret = Array{Float64}(undef,n,n,m,d)          # Output (2): Centroids
    times_ret = Array{Float64}(undef,n)                 # Output (3): Running times
    n_clusters = Array{Float64}(undef,n)                # Output (4): Number of clusters really considered

    for j in 1:m
        for p in 1:d
            k_cent[:,j,p] = copy(a[p][:,j])
        end
    end

	for j in n:-1:k+1
        texec = now()
        println("\nj is: $j / $texec")

        tini = time()

        if j == l

    		#Distances matrix designation
            if dm == 1
        		for i in 1:(l-1)
                    ## Create a centroids aux array to deal with the aggregation to
                    ## be tested (passing through each hour/cluster i)
                    k_cent_aux = copy(k_cent)
                    for i2 in 1:m, i3 in 1:d
                        k_cent_aux[i,i2,i3] = mean([k_cent[i,i2,i3],k_cent[i+1,i2,i3]])
                        k_cent_aux[i+1,i2,i3] = k_cent_aux[i,i2,i3]
                    end

                    ## Definition of what's the range of WD comparison (range_eval)

                    if j == 2
                        range_eval = i:i+1
                    elseif 1 < i < l - 1
                        range_eval = i-1:i+2
                    elseif i == 1
                        range_eval = i:i+2
                    elseif i == l-1
                        range_eval = i-1:i+1
                    else
                        range_eval = i-1:i
                    end

        			dist[i] = 2 * length(class[class[:] .== i]) *
        					   	  length(class[class[:] .== i+1]) /
        						  (length(class[class[:] .== i]) +
        						  length(class[class[:] .== i+1])) *
                                  ## Sum of WD from the n*d dimensions
                                  EMD_nD(k_cent[range_eval,:,:], k_cent_aux[range_eval,:,:])
        		end
            else
                for i in 1:(l-1)
        			dist[i] = 2 * length(class[class[:] .== i]) *
        					   	  length(class[class[:] .== i+1]) /
        						  (length(class[class[:] .== i]) +
        						  length(class[class[:] .== i+1])) *
        						  evaluate(Euclidean(),k_cent[i,:,:], k_cent[i+1,:,:])
        		end
            end
    		# Updating the last value for the max
    		dist[l] = maximum(dist[1:l])

    		# min: find the minimum distances
    		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

    		marker = zeros(n)					# To mark whenever the minimum occurs

    		##Mark as 1 whenever a minimum happens
    		for i in 1:(n-1)
    			if (class[i] in min_dist) && (class[i] != class[i+1])
    				marker[i+1] = 1
    			end
    		end

    		##Accounts the cumulative change in clusters order
    		c_change = 0						# Change in the number of clusters
    		for i in 1:n
    			c_change = c_change - marker[i]
    			marker[i] = c_change
    		end

    		##Update values
    		class[:] = class[:] + marker[:]

    		##Update centroids
    		for i in 1:l
                for j in 1:m
                    for p in 1:d
            			k_cent[i,j,p] = mean(a[p][class[:] .== i,j])
                    end
                end
    		end

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:,:] = copy(k_cent)

            l = l + Int(c_change)

        else

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:,:] = copy(k_cent)

        end

        n_clusters[j] = class_ret[j,n]
        Δt = time() - tini
        times_ret[j] = Δt

	end

    class_ret[n,:] = collect(1:n)
    for j in 1:m
        for p in 1:d
            k_cent_ret[n,:,j,p] = copy(a[p][:,j])
        end
    end
    n_clusters[n] = n
    n_clusters[1] = 1

    return class_ret, k_cent_ret, times_ret, n_clusters
end

### Clustering using representative days/weeks and medoids from Critical Points
function crit_RP(
    pD::Float64,
    pWS::Float64,
    D::Array{Float64},
    W::Array{Float64},
    S::Array{Float64},
    nodes::Int64,
    nRP::Int64,
    d::Array{Int64},
    pace::Float64,
    max_load::Vector{Float64},
    dem_inc::Vector{Float64}
    )
    #### Function made to select critical points accordingly to a minimal percentile
    #### for demand (pD) and a maximum percentile for W + S (pWS). The algorithm
    #### tries to find a maximum of representative periods nRP as specified in
    #### in the input.

    ### Computing the values using the input pD / pWS
    (range_D_ret, range_WS_ret, range_DWS_ret) = find_crit_DWS(pD,pWS,D.*max_load'.*dem_inc',W,S,nodes)
    crit_RP_ret = unique(d[findall(x -> x > 0,range_DWS_ret[3])])  # Number of days in the original

    ### Methodology for representative days

    ### Initialization
    pD_aux = copy(pD)                           # Percentile to find the critical points in Demand
    pWS_aux = copy(pWS)                         # Percentile to find the critical points in WS
    crit_RP_aux = copy(crit_RP_ret)             # Rep. periods using pD_aux and pWS_aux
    (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
    pD_hold = copy(pD_aux)
    pWS_hold = copy(pWS_aux)
    crit_RP_hold = copy(crit_RP_aux)

    # Test to see if a reduction on the nRP is needed
    if length(crit_RP_ret) >= nRP
        stop_aux = 0                            # Variable to stop the while loop
        while stop_aux < 1
            # Crop a quatile window equally reducing pWS and increasing pD
            pD_aux = pD_aux + pace              # Increase D percentile cut
            pWS_aux = pWS_aux - pace            # Decrease WS percentile cut
            # Find the ranges counting with the full demand
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            # Momentaneous RP
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            if length(crit_RP_aux) < nRP
                pD_aux = pD_aux - pace
                pWS_aux = pWS_aux + pace
                stop_aux = 1
            end
            # Updating the ranges after finding the first tight window for pD and pWS
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            # Update RP
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
        end

        # Adjustments in the case the pace was too high for the threshold
        stop_aux = 0
        while stop_aux < 1
            # Ok
            if length(crit_RP_aux) == nRP
                stop_aux = 1

            # Number of RP different than nRP
            else
                println(string("\nAdjustments needed when nRP is ",nRP," in ",now()))
                stop_aux2 = 0                           # Variable to control the next while loop
                n_int = 0                               # Number of iterations to control the time running the while loops
                low_pace = 0                            # Pace lower bound
                up_pace = copy(pace)                    # Pace upper bound
                pace_gap = up_pace - low_pace           # Pace difference to control the number of loops needed
                new_pace = pace/2                       # Updated pace
                while stop_aux2 < 1 && n_int < 1000
                    n_int += 1
                    println(string("\npace gap is: ",pace_gap))
                    pD_aux += new_pace
                    pWS_aux -= new_pace

                    #Reassess the ranges
                    (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
                    #Recalculate RP
                    crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])

                    if length(crit_RP_aux) == nRP
                        # Now we increase the criteria being necessary to match the number of days set by nRP
                        stop_aux2 = 1

                    else
                        println(" / Crit_days = ",length(crit_RP_aux))
                        # Back to the last step
                        pD_aux -= new_pace
                        pWS_aux += new_pace

                        if length(crit_RP_aux) > nRP
                            low_pace = copy(new_pace)
                            new_pace += (up_pace-new_pace)/2
                        else
                            up_pace = copy(new_pace)
                            new_pace -= (new_pace-low_pace)/2
                        end
                    end
                    pace_gap = up_pace - low_pace           # Pace difference to control the number of loops needed
                end
                stop_aux = 1
            end

        end

        if length(crit_RP_aux) != nRP
            println("\nNumber of days still not the same as nRP")
            pD_hold = pD_aux + up_pace
            pWS_hold = pWS_aux - up_pace
            (range_D_hold, range_WS_hold, range_DWS_hold) = find_crit_DWS(pD_hold,pWS_hold,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_hold = unique(d[findall(x -> x > 0,range_DWS_hold[3])])
        else
            pD_hold = copy(pD_aux)
            pWS_hold = copy(pWS_aux)
            (range_D_hold, range_WS_hold, range_DWS_hold) = find_crit_DWS(pD_hold,pWS_hold,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_hold = unique(d[findall(x -> x > 0,range_DWS_hold[3])])
        end

        ## Controlling if number of RP is higher than specified
        if length(crit_RP_hold) > nRP
            crit_RP_hold = copy(crit_RP_hold[1:nRP])
        end

        # Trying to decrease W+S percentile
        stop_aux = 0
        while stop_aux < 1
            # Update pD
            pD_aux = pD_aux + pace
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            if length(crit_RP_aux) != length(crit_RP_hold)
                pD_aux = pD_aux - pace
                stop_aux = 1
            end
        end

        # Trying to increase the Demand percentile
        stop_aux = 0
        while stop_aux < 1
            # Update pWS
            pWS_aux = pWS_aux - pace
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            if length(crit_RP_aux) != length(crit_RP_hold)
                pWS_aux = pWS_aux + pace
                stop_aux = 1
            end
        end
    end

    (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
    crit_RP_aux_ret = unique(d[findall(x -> x > 0,range_DWS_aux[3])])

    # Return with the percentiles used
    range_D_aux_ret = range_D_aux
    range_WS_aux_ret = range_WS_aux
    range_DWS_aux_ret = range_DWS_aux

    if crit_RP_hold != crit_RP_aux_ret
        println("\n Diferent RP! Check code. \n")
    end

    return range_D_ret, range_WS_ret, range_DWS_ret, crit_RP_ret,
    range_D_aux_ret, range_WS_aux_ret, range_DWS_aux_ret, crit_RP_aux_ret
end

### Clustering using representative days/weeks and medoids
function RDC3D_new(k::Int64,
	a₁::Array{Float64},
    a₂::Array{Float64},
    a₃::Array{Float64})

    ## aᵢ has the size of p x q x m where m is the number of nodes, q is the
    ## size in hours of the representative set (e.g. if representative day then
    ## q = 24h) and p is the number of representative periods in a year (e.g.
    ## if representative day, p = 365) in a way that p x q = 8760h.


    if size(a₁) != size(a₂) || size(a₁) != size(a₃)
        println("Different sizes input")
        return
    end


	### Initialization
	p = length(a₁[:,1,1])       # Total number of hours per repr. period
    q = length(a₁[1,:,1])       # Total number of periods in a year

    if p*q != 8760
        println("Number of periods do not represent one year")
        return
    end

    m = length(a₁[1,1,:])                               # Number of nodes
	class = collect(1:q)                                # Array of classes (n)
    k_cent = zeros(p,q,m,3)                             # Centroids array [hour,day,node,DWS]
	dist = zeros(q)                                     # Array of distances (n)
	l = q                                               # Number of clusters (in this case, days) used
	counter = 0                                         # Control the number of iterations
    class_ret = Array{Float64}(undef,q,q)               # Output (1): Classes
    k_cent_ret = Array{Float64}(undef,q,p,q,m,3)        # Output (2): Centroids / Medoids
    times_ret = Array{Float64}(undef,q)                 # Output (3): Running times
    n_clusters = Array{Float64}(undef,q)                # Output (4): Number of clusters really considered

    for i in 1:p
        for j in 1:q
            for l in 1:m
                k_cent[i,j,l,:] = copy([a₁[i,j,l,:] a₂[i,j,l,:] a₃[i,j,l,:]])
            end
        end
    end

    for j in q:-1:k+1
        texec = now()
        println("Number of representative days: $j / $texec")
        tini = time()

        if j == l

            #Distances matrix designation
            for i in 1:(l-1)
                dist[i] = evaluate(Euclidean(),k_cent[:,i,:,:], k_cent[:,i+1,:,:])
            end

    		# Updating the last value for the max
    		dist[l] = maximum(dist[1:l-1])

    		# min: find the minimum distances
    		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

    		marker = zeros(q)					# To mark whenever the minimum occurs

    		##Mark as 1 whenever a minimum happens
    		for i in 1:(q-1)
    			if (class[i] in min_dist) && (class[i] != class[i+1])
    				marker[i+1] = 1
    			end
    		end

    		##Accounts the cumulative change in clusters order
    		c_change = 0						# Change in the number of clusters
    		for i in 1:q
    			c_change = c_change - marker[i]
    			marker[i] = c_change
    		end

    		##Update values
    		class[:] = class[:] + marker[:]

    		##Update centroids
    		for i in 1:p              # Hour
                for k in 1:q          # Day
                    for o in 1:m      # Nodes
            			k_cent[i,k,o,1] = mean(a₁[i,class[:] .== k,o])
                        k_cent[i,k,o,2] = mean(a₂[i,class[:] .== k,o])
                        k_cent[i,k,o,3] = mean(a₃[i,class[:] .== k,o])
                    end
                end
    		end

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:,:,:] = copy(k_cent)

            l = l + Int(c_change)

        else

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:,:,:] = copy(k_cent)

        end

        n_clusters[j] = class_ret[j,q]
        Δt = time() - tini
        times_ret[j] = Δt

	end

    ##Classes with full resolution
    class_ret[q,:] = collect(1:q)

    ##Centroids with full resolution
    for i in 1:p
        for j in 1:q
            for o in 1:m
                k_cent_ret[q,i,j,o,:] = copy([a₁[i,j,o,:] a₂[i,j,o,:] a₃[i,j,o,:]])
            end
        end
    end

    n_clusters[q] = q
    n_clusters[1] = 1

    return class_ret, k_cent_ret, times_ret, n_clusters
end

### Compute Wass. Dist in 2D series
function wassdist2D(from::Array{Float64},
    to::Array{Float64})

    #### NOTE: uncomment this to check if the distributions' mass match
    # for d in 1:2
    #     if sum(from[:,d]) != sum(to[:,d])
    #         println("\n Arrays with differente mass in dimension $d \n")
    #         # return
    #     end
    # end
    ##__________________________________________________________________________
    ##### Probabilistic Distance based on Wasserstein distance
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1])

    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(CPLEX.Optimizer)

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution
    K = collect(1:2)    		# Number of dimensions

    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[i = 1:I[end], j = 1:J[end], k = 1:K[end]] >= 0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I, k in K],  P_i[i,k] == sum(eta[i,j,k] for j in J))
    @constraint(model,[j in J, k in K],  P_j[j,k] == sum(eta[i,j,k] for i in I))

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    @objective(model, Min, sum(abs(i-j)*eta[i,j,k] for i in I, j in J, k in K))

    # status = @suppress solve(model)

    #Get the values of costs against number of clusters
    out = objective_value(model)

    return out
end

# Type for arcs (i,j)
struct Arc
    i::Int64  # Start node of arc (i,j)
    j::Int64  # End node of arc (i,j)
end

### Compute Wass. Dist in nD series
function wassdist_new(from::Array{Float64},
    to::Array{Float64},
    dist::Int64)

    ## Algorithm to compute the Wasserstein distance from a vector  called
    # "from" and another vector named "to". The distance between active nodes
    # cannot surpass the limit established by "dist"

    ##__________________________________________________________________________
    ##### Probabilistic Distance based on Wasserstein distance
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1])

    if dist > (n-1)
        println("Inconsistent difference between time steps")
        return
    end

    ### Building the diagonal matrix
    from_to = Int.(zeros(n,n))
    for i in 1:n
        for j in 1:n
            if abs(i-j) <= dist
                from_to[i,j] = 1
            end
        end
    end

    # Counting the number of elements
    counter = sum(from_to)

    # Building the matrix with active elements
    aux1 = from_to[[i for i = 1:n*n]].*collect(1:n*n)
    active = aux1[aux1 .!= 0]

    # Creating i -> j structure
    inode = Int.(zeros(counter))
    jnode = Int.(zeros(counter))

    # Filling (i,j) arcs
    for i in 1:counter
        jnode[i] = div(active[i]-1,n)+1
        inode[i] = rem(active[i]-1,n)+1
    end

    # Construct arcs set
    A = Arc.(inode, jnode)

    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(Gurobi.Optimizer)  #Using dual simplex (Algorithm 1)

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution

    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[A]>=0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I], P_i[i] == sum(eta[a] for a in A if a.i == i))
    @constraint(model,[j in J], P_j[j] == sum(eta[a] for a in A if a.j == j))

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    costs = zeros(length(A))

    for a in 1:length(A)
        costs[a] = abs(A[a].i-A[a].j)
    end

    @objective(model, Min, sum(costs[a]*eta[Arc(A[a].i,A[a].j)] for a in 1:length(A)))

    status = @suppress optimize!(model)

    #Get the values of costs against number of clusters
    out = objective_value(model)

    return out
end

### Compute Wass. Dist in nD series (final version)
function wassdist_vf(from::Array{Float64},
    to::Array{Float64},
    dist::Int64)

    ## Algorithm to compute the Wasserstein distance from a vector  called
    # "from" and another vector named "to". The distance between active nodes
    # cannot surpass the limit established by "dist"

    ##__________________________________________________________________________
    ##### Probabilistic Distance based on Wasserstein distance
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1])

    if dist > (n-1)
        println("Inconsistent difference between time steps")
        return
    end

    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(Gurobi.Optimizer)  #Using dual simplex (Algorithm 1)

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution

    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[i in I, j in J ; abs(i-j) <= dist]>=0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I], P_i[i] == sum(eta[i,j] for j in J if abs(i-j)<=dist))
    @constraint(model,[j in J], P_j[j] == sum(eta[i,j] for i in I if abs(i-j)<=dist))

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    @objective(model, Min, sum(abs(i-j)*eta[i,j] for i in I, j in J if abs(i-j)<=dist))

    status = @suppress optimize!(model)

    #Get the values of costs against number of clusters
    out = objective_value(model)

    return out
end

function EMD_effic(from::Array{Float64},
    to::Array{Float64})

    ## Algorithm to compute the Wasserstein distance (or Earth Mover's Distance)
    # from a vector called "from" and another vector named "to". The distance
    # between active nodes cannot surpass the limit established by "dist"

    ##__________________________________________________________________________
    ##### Probabilistic Distance computation based on the algorithm proposed
    ## by (Ling and Okada, 2007: An Efficient Earth Mover’s Distance Algorithm
    ## for Robust Histogram Comparison)
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1])


    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(Gurobi.Optimizer)  #Using dual simplex (Algorithm 1)

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution


    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[i in I, j in J ; abs(i-j) <= 1]>=0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I],
    sum(eta[i,j] for j in J if abs(i-j)<=1) - sum(eta[j,i] for j in J if abs(i-j)<=1)
    == from[i] - to[i])

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    @objective(model, Min, sum(abs(i-j)*eta[i,j] for i in I, j in J if abs(i-j)<=1))

    status = @suppress optimize!(model)

    #Get the values of costs against number of clusters
    out = objective_value(model)

    return out
end

### Compute Wass. Dist in 3D series (efficient version)
function EMD_3D(from::Array{Float64},
    to::Array{Float64})

    ## Algorithm to compute the adaptive Wasserstein distance (or Earth Mover's
    # Distance) from a vector called "from" (which represents the original
    # series) and another vector named "to" (adapted series).

    ##__________________________________________________________________________
    ##### Probabilistic Distance computation based on the algorithm proposed
    ## by (Ling and Okada, 2007: An Efficient Earth Mover’s Distance Algorithm
    ## for Robust Histogram Comparison)
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1,1])             # (number of hours)
    m = length(from[1,:,1])             # (number of nodes)
    s = length(from[1,1,:])             # (number of series - DWS)

    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(Gurobi.Optimizer)  #Using dual simplex (Algorithm 1)

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution
    K = collect(1:m)            # Nodes
    R = collect(1:s)            # Dimensions (D,W,S)

    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[i in I, j in J, k in K, r in R ; abs(i-j) <= 1]>=0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I, k in K, r in R],
    sum(eta[i,j,k,r] for j in J if abs(i-j)<=1) - sum(eta[j,i,k,r] for j in J if abs(i-j)<=1)
    == from[i,k,r] - to[i,k,r])

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    @objective(model, Min, sum(abs(i-j)*eta[i,j,k,r] for i in I, j in J, k in K, r in R
    if abs(i-j)<=1))

    status = @suppress optimize!(model)

    #Get the values of costs against number of clusters
    out = objective_value(model)

    return out
end

### Compute Wass. Dist in nD series (auxiliar version)
function EMD_nD(from::Array{Float64},
    to::Array{Float64})

    ## Algorithm to compute the adaptive Wasserstein distance (or Earth Mover's
    # Distance) from a vector called "from" (which represents the original
    # series) and another vector named "to" (adapted series).

    ##__________________________________________________________________________
    ##### Probabilistic Distance computation based on the algorithm proposed
    ## by (Ling and Okada, 2007: An Efficient Earth Mover’s Distance Algorithm
    ## for Robust Histogram Comparison)
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1,1])             # (number of hours)
    m = length(from[1,:,1])             # (number of nodes)
    s = length(from[1,1,:])             # (number of series - DWS)

    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(Gurobi.Optimizer)  #Using dual simplex (Algorithm 1)

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution
    K = collect(1:m)            # Nodes
    R = collect(1:s)            # Dimensions (D,W,S)

    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[i in I, j in J, k in K, r in R ; abs(i-j) <= 1]>=0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I, k in K, r in R],
    sum(eta[i,j,k,r] for j in J if abs(i-j)<=1) - sum(eta[j,i,k,r] for j in J if abs(i-j)<=1)
    == from[i,k,r] - to[i,k,r])

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    @objective(model, Min, sum(abs(i-j)*eta[i,j,k,r] for i in I, j in J, k in K, r in R
    if abs(i-j)<=1))

    status = @suppress optimize!(model)

    #Get the values of costs against number of clusters
    out = objective_value(model)

    return out
end

function pm_wass_computation(D::Array{Float64},
    W::Array{Float64},
    S::Array{Float64},
    m::Array{Int64},
    pm_wassd_P::Vector{Any},
    pm_wassd_W::Vector{Any},
    pm_P_centroids::Vector{Any},
    pm_W_centroids::Vector{Any},
    step::Int64,
    dist::Int64,
    month_from::Int64,
    month_to::Int64)

    for k in month_from:month_to
        for i in 1:step:length(D[m.==k])
            pm_wassd_P[k][i,1] = wassdist_new(D[m.==k],pm_P_centroids[k][i,:,1],dist)
            pm_wassd_P[k][i,2] = wassdist_new(W[m.==k],pm_P_centroids[k][i,:,2],dist)
            pm_wassd_P[k][i,3] = wassdist_new(S[m.==k],pm_P_centroids[k][i,:,3],dist)
            pm_wassd_W[k][i,1] = wassdist_new(D[m.==k],pm_W_centroids[k][i,:,1],dist)
            pm_wassd_W[k][i,2] = wassdist_new(W[m.==k],pm_W_centroids[k][i,:,2],dist)
            pm_wassd_W[k][i,3] = wassdist_new(S[m.==k],pm_W_centroids[k][i,:,3],dist)

            if pm_wassd_P[k][i,1] > 0 && pm_wassd_P[k][i,2] > 0 && pm_wassd_P[k][i,3] > 0
                pm_wassd_P_total[k][i] = pm_wassd_P[k][i,1] + pm_wassd_P[k][i,2] + pm_wassd_P[k][i,3]
            end

            if pm_wassd_W[k][i,1] > 0 && pm_wassd_W[k][i,2] > 0 && pm_wassd_W[k][i,3] > 0
                pm_wassd_W_total[k][i] = pm_wassd_W[k][i,1] + pm_wassd_W[k][i,2] + pm_wassd_W[k][i,3]
            end
        end
    end
end

function EMD_computation_n_nodes(nodes::Int64,
    Cluss::UnitRange{Int64},
    D::Array{Float64},
    W::Array{Float64},
    S::Array{Float64},
    P_centroids_DWS::Array{Float64})

    wassd_P = zeros(length(Cluss))
    wassd_aux = zeros(length(Cluss),3,3)

    for i in Cluss
        # println("Cluster level is: ", i," at ", now(), "\n")
        for j in 1:nodes
            wassd_aux[i,j,1] = EMD_effic(D[:,j],P_centroids_DWS[i,:,j,1])
            wassd_aux[i,j,2] = EMD_effic(W[:,j],P_centroids_DWS[i,:,j,2])
            wassd_aux[i,j,3] = EMD_effic(S[:,j],P_centroids_DWS[i,:,j,3])
        end
        if sum(wassd_aux[:,:,1]) > 0 &&
            sum(wassd_aux[:,:,2]) > 0 &&
            sum(wassd_aux[:,:,3]) > 0
            wassd_P[i] = sum(wassd_aux[i,:,:])
        end
    end

    return wassd_P, wassd_aux
end


##### Cluster aggregation function
function aggreg(a₁::Array{Float64},
    sizes::Array{Int64}, marker::Array{Int64})

    #### Function to aggregate clusters, normally used in hierarchical
    #### clustering. It pass through a vector a₁ merging the state (i)
    #### with (i + 1) whenever there's a marker

	## Initialization
    n = length(a₁)                      # Size of vectors
    class = collect(1:n)                # Vector to mark the new classes
    new_a₁ = copy(a₁)                   # New vector to be formed after aggregation
	aux = zeros(n)

    ## Error control
    if length(a₁) != length(marker) || length(a₁) != length(sizes)
        println("\n Different sizes \n")
    	return
	end

	for i in 1:n
		if !(marker[i] == 0 || marker[i] == 1)
			println("\n Inconsistent marker vector \n")
			return
		end
	end

    ##Accounts the cumulative change in clusters order
    c_change = 0						# Change in the number of clusters
    for i in 2:n
        c_change = c_change - marker[i-1]
	    aux[i] = c_change
    end

    ## Update values
    class[:] = class[:] + aux[:]

    ## Calculate the pondered average
    for i in 1:(n-sum(marker[:]))
        new_a₁[class[:].==i] .= a₁[class[:].==i]'*sizes[class[:].==i]/sum(sizes[class[:].==i])
    end

    return new_a₁, class
end

function cond2D_short(k::Int64,
	a₁::Vector{Float64},
    a₂::Vector{Float64})

	## Clustering modelling inpired by the article named "Chronological
    ## Time-Period Clustering for Optimal Capacity Expansion Planning
    ## With Storage" by Salvador Pineda & Juan Morales (2018)

    if length(a₁) != length(a₂)
        println("Different sizes input")
        return
    end

	### Initialization
	n = length(a₁)						# Size of Data
	class = collect(1:n)				# Array of classes (n)
	k_cent = copy([a₁ a₂])              # Centroids array
	dist = zeros(n)						# Array of distances (n)
	l = length(a₁)						# Number of clusters used
	δ = 10^-5                           # Small increment
    sizes = ones(Int64,n)               # Sizes of each cluster
    class_ret = Array{Float64}(undef,n,n)  # Output (1)
    k_cent_ret = Array{Float64}(undef,n,n,2) # Output (2)
    times_ret = Array{Float64}(undef,n)     #Output (3)

	for j in n:-1:k+1
        # texec = now()
        # println("j = $j / Time: $texec \n")

        tini = time()

        if j == l

            #Distances matrix designation
            for i in 1:(l-1)
                new_class = copy(class)                 # Aux
                new_k_cent = copy(k_cent)               # Aux
                new_k_class = collect(1:l)              # Aux

                ##Create a vector with a aggregation marker in i
                aux = zeros(Int64,l)
                aux[i] = 1

                ##Create momentary classes and k_centers
                new_k_class = aggreg(k_cent[1:l,1],sizes[1:l],aux)[2]
                change_class = new_k_class - collect(1:l)
                new_class = class + change_class[class]

                for i in 1:l
                	new_k_cent[i,1] = mean(a₁[new_class[:] .== i])
                    new_k_cent[i,2] = mean(a₂[new_class[:] .== i])
                end

                dist[i] = wassdist2D([a₁ a₂],
                new_k_cent[new_class,:])
            end

    		# Updating the last value for the max
    		dist[l] = maximum(dist[1:l]) + δ

    		# min: find the minimum distances
    		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

    		marker = zeros(n)					# To mark whenever the minimum occurs

    		##Mark as 1 whenever a minimum happens
    		for i in 1:(n-1)
    			if (class[i] in min_dist) && (class[i] != class[i+1])
    				marker[i+1] = 1
    			end
    		end

            marker_p = marker[marker.==1]

    		##Accounts the cumulative change in clusters order
    		c_change = 0						# Change in the number of clusters
    		for i in 1:n
    			c_change = c_change - marker[i]
    			marker[i] = c_change
    		end

    		##Update values
    		class[:] = class[:] + marker[:]

    		##Update centroids
    		for i in 1:l
    			k_cent[i,1] = mean(a₁[class[:] .== i])
                k_cent[i,2] = mean(a₂[class[:] .== i])
                sizes[i] = sum(class[:] .== i)
    		end

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:] = copy(k_cent)

    		l = l + Int(c_change)

        else

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:] = copy(k_cent)

        end
        Δt = time() - tini
        times_ret[j] = Δt
    end

    class_ret[n,:] = collect(1:n)
    k_cent_ret[n,:,:] = copy([a₁ a₂])

	return class_ret, k_cent_ret, times_ret
end

function cond2D_v2(k::Int64,
	a₁::Vector{Float64},
    a₂::Vector{Float64})

	## Clustering modelling inspired by the article named "Chronological
    ## Time-Period Clustering for Optimal Capacity Expansion Planning
    ## With Storage" by Salvador Pineda & Juan Morales (2018)

    if length(a₁) != length(a₂)
        println("Different sizes input")
        return
    end

	### Initialization
	n = length(a₁)						# Size of Data
	class = collect(1:n)				# Array of classes (n)
	k_cent = copy([a₁ a₂])              # Centroids array
	dist = zeros(n)						# Array of distances (n)
	l = length(a₁)						# Number of clusters used
	δ = 10^-5                           # Small increment
    sizes = ones(Int64,n)               # Sizes of each cluster
    class_ret = Array{Float64}(undef,n,n)  # Output (1)
    k_cent_ret = Array{Float64}(undef,n,n,2) # Output (2)
    times_ret = Array{Float64}(undef,n)     #Output (3)

	for j in n:-1:k+1

        tini = time()

        if j == l

            #Distances matrix designation
            for i in 1:(l-1)
                new_class = copy(class)                 # Aux
                new_k_cent = copy(k_cent)               # Aux
                new_k_class = collect(1:l)              # Aux

                ##Create a vector with a aggregation marker in i
                aux = zeros(Int64,l)
                aux[i] = 1

                ##Create momentary classes and k_centers
                new_k_class = aggreg(k_cent[1:l,1],sizes[1:l],aux)[2]
                change_class = new_k_class - collect(1:l)
                new_class = class + change_class[class]

                for i in 1:l
                	new_k_cent[i,1] = mean(a₁[new_class[:] .== i])
                    new_k_cent[i,2] = mean(a₂[new_class[:] .== i])
                end

                dist[i] = wassdist2D([a₁ a₂],
                new_k_cent[new_class,:])
            end

    		# Updating the last value for the max
    		dist[l] = maximum(dist[1:l]) + δ

    		# min: find the minimum distances
    		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

    		marker = zeros(n)					# To mark whenever the minimum occurs

    		##Mark as 1 whenever a minimum happens
    		for i in 1:(n-1)
    			if (class[i] in min_dist) && (class[i] != class[i+1])
    				marker[i+1] = 1
    			end
    		end

            marker_p = marker[marker.==1]

    		##Accounts the cumulative change in clusters order
    		c_change = 0						# Change in the number of clusters
    		for i in 1:n
    			c_change = c_change - marker[i]
    			marker[i] = c_change
    		end

    		##Update values
    		class[:] = class[:] + marker[:]

    		##Update centroids
    		for i in 1:l
    			k_cent[i,1] = mean(a₁[class[:] .== i])
                k_cent[i,2] = mean(a₂[class[:] .== i])
                sizes[i] = sum(class[:] .== i)
    		end

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:] = copy(k_cent)

    		l = l + Int(c_change)

        else

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:] = copy(k_cent)

        end
        Δt = time() - tini
        times_ret[j] = Δt
    end

    class_ret[n,:] = collect(1:n)
    k_cent_ret[n,:,:] = copy([a₁ a₂])

	return class_ret, k_cent_ret, times_ret
end

function cond3D_new(k::Int64,
	a₁::Vector{Float64},
    a₂::Vector{Float64},
    a₃::Vector{Float64})

	## Clustering modelling inspired by the article named "Chronological
    ## Time-Period Clustering for Optimal Capacity Expansion Planning
    ## With Storage" by Salvador Pineda & Juan Morales (2018)

    if length(a₁) != length(a₂)
        println("Different sizes input between a₁ and a₂")
        return
    elseif length(a₁) != length(a₃)
        println("Different sizes input between a₁ and a₃")
        return
    end

	### Initialization
    num_dims = 3
    n = length(a₁)						# Size of Data
	class = collect(1:n)				# Array of classes (n)
	k_cent = copy([a₁ a₂ a₃])           # Centroids array
	dist = zeros(n)						# Array of distances (n)
	l = length(a₁)						# Number of clusters used
	δ = 10^-5                           # Small increment
    counter = 0							        # Control the number of iterations
    sizes = ones(Int64,n)               # Sizes of each cluster
    class_ret = Array{Float64}(undef,n,n)  # Output (1)
    k_cent_ret = Array{Float64}(undef,n,n,num_dims) # Output (2)
    times_ret = Array{Float64}(undef,n)     #Output (3)

	for j in n:-1:k+1
        texec = now()
        # println("j is: $j / $texec")

        tini = time()

        if j == l

            #Distances matrix designation
            for i in 1:(l-1)
                new_class = copy(class)                 # Aux
                new_k_cent = copy(k_cent)               # Aux
                new_k_class = collect(1:l)              # Aux

                ##Create a vector with an aggregation marker in i
                aux = zeros(Int64,l)
                aux[i] = 1

                ##Create momentary classes and k_centers
                new_k_class = aggreg(k_cent[1:l,1],sizes[1:l],aux)[2]
                change_class = new_k_class - collect(1:l)
                new_class = class + change_class[class]

                for i in 1:l
                	new_k_cent[i,1] = mean(a₁[new_class[:] .== i])
                    new_k_cent[i,2] = mean(a₂[new_class[:] .== i])
                    new_k_cent[i,3] = mean(a₃[new_class[:] .== i])
                end

                if length(a₁)>= 47
                    dist[i] = wassdist_new([a₁ a₂ a₃],
                    new_k_cent[new_class,:],47)
                else
                    dist[i] = wassdist_new([a₁ a₂ a₃],
                    new_k_cent[new_class,:],length(a₁)-1)
                end
            end

    		# Updating the last value for the max
    		dist[l] = maximum(dist[1:l]) + δ

    		# min: find the minimum distances
    		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

    		marker = zeros(n)					# To mark whenever the minimum occurs

    		##Mark as 1 whenever a minimum happens
    		for i in 1:(n-1)
    			if (class[i] in min_dist) && (class[i] != class[i+1])
    				marker[i+1] = 1
    			end
    		end

            marker_p = marker[marker.==1]

    		##Accounts the cumulative change in clusters order
    		c_change = 0						# Change in the number of clusters
    		for i in 1:n
    			c_change = c_change - marker[i]
    			marker[i] = c_change
    		end

    		##Update values
    		class[:] = class[:] + marker[:]

    		##Update centroids
    		for i in 1:l
    			k_cent[i,1] = mean(a₁[class[:] .== i])
                k_cent[i,2] = mean(a₂[class[:] .== i])
                k_cent[i,3] = mean(a₃[class[:] .== i])
                sizes[i] = sum(class[:] .== i)
    		end

            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:] = copy(k_cent)

    		l = l + Int.(c_change)

        else

            #Branch for unused clusters
            class_ret[j-1,:] = class
            k_cent_ret[j-1,:,:] = copy(k_cent)

        end
        Δt = time() - tini
        times_ret[j] = Δt
    end

    class_ret[n,:] = collect(1:n)
    k_cent_ret[n,:,:] = copy([a₁ a₂ a₃])

	return class_ret, k_cent_ret, times_ret
end

function red_clust(h_min::Int64,pm_P_clusters::Vector{Any},mode::Int64)
    ## Mode: 1-Dem; 2-Wind; 3-Solar
    red_cent = Array{Float64}(undef,h_min,h_min*12)

    for i in 1:h_min
        aux = zeros(Int64,12)

        for k in 1:12
            aux[k] = i - length(unique(pm_P_clusters[k][i,:]))
        end

        size_inst_P = sum(length(unique(pm_P_clusters[k][i,:])) for k in 1:12)

        red_cent[i,1:12i] = [

        append!(pm_P_centroids[1][i,
        findfirst.(isequal.(unique(pm_P_clusters[1][i,:])),
        [pm_P_clusters[1][i,:]]),mode],zeros(Float64,aux[1]))

        append!(pm_P_centroids[2][i,
        findfirst.(isequal.(unique(pm_P_clusters[2][i,:])),
        [pm_P_clusters[2][i,:]]),mode],zeros(Float64,aux[2]))

        append!(pm_P_centroids[3][i,
        findfirst.(isequal.(unique(pm_P_clusters[3][i,:])),
        [pm_P_clusters[3][i,:]]),mode],zeros(Float64,aux[3]))

        append!(pm_P_centroids[4][i,
        findfirst.(isequal.(unique(pm_P_clusters[4][i,:])),
        [pm_P_clusters[4][i,:]]),mode],zeros(Float64,aux[4]))

        append!(pm_P_centroids[5][i,
        findfirst.(isequal.(unique(pm_P_clusters[5][i,:])),
        [pm_P_clusters[5][i,:]]),mode],zeros(Float64,aux[5]))

        append!(pm_P_centroids[6][i,
        findfirst.(isequal.(unique(pm_P_clusters[6][i,:])),
        [pm_P_clusters[6][i,:]]),mode],zeros(Float64,aux[6]))

        append!(pm_P_centroids[7][i,
        findfirst.(isequal.(unique(pm_P_clusters[7][i,:])),
        [pm_P_clusters[7][i,:]]),mode],zeros(Float64,aux[7]))

        append!(pm_P_centroids[8][i,
        findfirst.(isequal.(unique(pm_P_clusters[8][i,:])),
        [pm_P_clusters[8][i,:]]),mode],zeros(Float64,aux[8]))

        append!(pm_P_centroids[9][i,
        findfirst.(isequal.(unique(pm_P_clusters[9][i,:])),
        [pm_P_clusters[9][i,:]]),mode],zeros(Float64,aux[9]))

        append!(pm_P_centroids[10][i,
        findfirst.(isequal.(unique(pm_P_clusters[10][i,:])),
        [pm_P_clusters[10][i,:]]),mode],zeros(Float64,aux[10]))

        append!(pm_P_centroids[11][i,
        findfirst.(isequal.(unique(pm_P_clusters[11][i,:])),
        [pm_P_clusters[11][i,:]]),mode],zeros(Float64,aux[11]))

        append!(pm_P_centroids[12][i,
        findfirst.(isequal.(unique(pm_P_clusters[12][i,:])),
        [pm_P_clusters[12][i,:]]),mode],zeros(Float64,aux[12]))

        ]
    end

    return red_cent
end

function RMSE(from::Array{Float64},
    to::Array{Float64})

    ##__________________________________________________________________________
    ##### Root Mean Squared between two series
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1])
    accum = 0

    accum = sum(sqrt.(((from - to).^2)./n))

    return accum
end

function CTEM_int(
    nodes::Int64,
    lines::Vector{Array{Int64,1}},
    LT::Vector{Int64},
    K::Vector{Float64},
    KA::Array{Float64},
    CO::Array{Float64},
    CS::Float64,
    GCN::Array{Float64},
    SC::Float64,
    KIT::Array{Float64},
    KT::Float64,
    T_effic::Float64,
    KIS::Float64,
    KS::Float64,
    S_effic_c::Array{Float64},
    S_effic_d::Array{Float64},
    S_0::Array{Float64},
    RES::Float64,
    D::Array{Float64},
    W_avail::Array{Float64},
    S_avail::Array{Float64},
    max_load::Vector{Float64},
    dem_inc::Vector{Float64}
    )

    ####   Modeling the Multi-nodal CTEM for 5 different technologies (int)  ###
    ## Data to be used:
    # RES: level of renewables needed into the system
    # nodes: Number of nodes
    # lines: possible lines connections
    # D: Demand
    # W_avail: Wind availability
    # S_avail: Solar availability

    #______________________________________________________________________________#
    ############################    Sets    ########################################
    #______________________________________________________________________________#

    num_h = length(D[:,1])              # Number of hours

    H = collect(1:num_h)                # Total hours (Integral approach)
    T = collect(1:5)                    # Technologies
    N = collect(1:nodes)                # Nodes set
    L = copy(lines)                     # Lines set

    ## Technologies
    #   1: Wind
    #   2: Solar
    #   3: Coal
    #   4: Gas
    #   5: Oil
    ##

    #______________________________________________________________________________#
    ############################    Parameters    ##################################
    #______________________________________________________________________________#

    Dem = zeros(H[end], N[end])

    ## Full-time resolution Demand for 2030
    for n in N
        Dem[:,n] = dem_inc[n].*D[:,n].*max_load[n]             # Integral Demand in each node (MWh)
    end

    ############################################################################
    ###################    MULTI-NODAL MODEL (integral)    #####################
    ############################################################################

    #__________________________________________________________________________#
    ###########################    Variables    ################################
    #__________________________________________________________________________#
    # opt_model = Model(Gurobi.Optimizer)
    opt_model = Model(Gurobi.Optimizer)

    @variable(opt_model, p[t in T, ts in H, n in N]>=0)    # Generation level in each hour by t [MW]
    @variable(opt_model, p_bar[t in T, n in N]>=0)         # Additional generation capac. of t [MW]
    @variable(opt_model, σ[ts in H, n in N]>=0)            # Energy shedding in ts [MW]
    @variable(opt_model, f[ts in H, l in L])               # Transmission flow in each hour and line [MW]
    @variable(opt_model, exp[ts in H, l in L]>=0)          # Positive transmission flow in each and line [MW]
    @variable(opt_model, f_bar[l in L]>=0)                 # Additional transmission capac. of t [MW]
    @variable(opt_model, s[ts in H, n in N] >= 0)          # Storage level in each hour [MWh]
    @variable(opt_model, s_bar[n in N]>=0)                 # Storage capac. of t [MWh]
    @variable(opt_model, charg[ts in H, n in N] >= 0)      # Charging in each hour [MW]
    @variable(opt_model, discharg[ts in H, n in N] >= 0)   # Discharging in each hour [MW]

    #______________________________________________________________________________#
    ########################    Objective Function    ##############################
    #______________________________________________________________________________#

    @objective(opt_model, Min, sum((K[t]+KA[t])*p_bar[t,n] for t in T, n in N) +
                               sum(CO[t]*p[t,ts,n] for t in T, ts in H, n in N) +
                               sum(CS*σ[ts,n] for ts in H,n in N) +
                               sum((KIT[lo] + KAT)*f_bar[L[lo]] for lo in 1:length(L)) +
                               sum(KT*exp[ts,l] for ts in H, l in L) +
                               sum(KIS*s_bar[n] for n in N) +
                               sum(KS*(charg[ts,n] + discharg[ts,n]) for ts in H, n in N))
    #
    #______________________________________________________________________________#
    ############################    Constraints    #################################
    #______________________________________________________________________________#

    # #1a) Energy balance (using charging/discharging)
    # @constraint(opt_model,[ts in H,n in N],
    # sum(p[t,ts,n] for t in T) + σ[ts,n] -
    # sum(f[ts,l] for l in L if l[1] .== n) +
    # sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    # S_effic_c[n]*charg[ts,n] +
    # S_effic_d[n]*discharg[ts,n]
    # == Dem[ts,n])

    #1a) Energy balance (ts > 1)
    @constraint(opt_model,[ts in H[H.>1],n in N],
    sum(p[t,ts,n] for t in T) + σ[ts,n] -
    sum(f[ts,l] for l in L if l[1] .== n) +
    sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    S_effic_c[n]*(s[ts,n] - s[ts-1,n])
    == Dem[ts,n])

    #1b) Energy balance (ts = 1)
    @constraint(opt_model,[ts in [1],n in N],
    sum(p[t,ts,n] for t in T) + σ[ts,n] -
    sum(f[ts,l] for l in L if l[1] .== n) +
    sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    S_effic_c[n]*(s[ts,n] - S_0[n])
    == Dem[ts,n])

    #2) Wind generation capacity
    @constraint(opt_model,[t in [1],ts in H,n in N],
    p[t,ts,n] <= W_avail[ts,n].*(GCN[t,n] + p_bar[t,n]))

    #3) Solar generation capacity
    @constraint(opt_model,[t in [2],ts in H,n in N],
    p[t,ts,n] <= S_avail[ts,n].*(GCN[t,n] + p_bar[t,n]))

    #4) NRES capacity
    @constraint(opt_model,[t in [3,4,5],ts in H,n in N],
    p[t,ts,n] <= (GCN[t,n] + p_bar[t,n]))

    #5) Minimum RES
    @constraint(opt_model,
    sum(p[t,ts,n] for t in [1,2],ts in H,n in N) >=
    RES*sum(p[t,ts,n] for t in T,ts in H,n in N))

    # @constraint(opt_model,
    # sum(p[t,ts,n] for t in [1,2],ts in H,n in N) >=
    # sum(RES3*Dem[ts,n] for ts in H, n in N))

    # #6) Budget?
    # @constraint(opt_model, sum((K[t]+KA[t]).*p_bar[t] for t in T, ts in TS) <= B)

    #7) Shedding cap
    @constraint(opt_model,[ts in H, n in N],
    σ[ts,n] <= SC*Dem[ts,n])

    #8) Transmission cap
    @constraint(opt_model,[ts in H, l in L],
    f[ts,l] <= f_bar[l])
    @constraint(opt_model,[ts in H, l in L],
    -f_bar[l] <= f[ts,l])

    #9a) Charge / Discharge (ts > 1)
    # @constraint(opt_model, [ts in H[H.>1], n in N],
    # discharg[ts, n] >= -s[ts, n] + s[ts - 1, n])
    # @constraint(opt_model, [ts in H[H.>1], n in N],
    # charg[ts, n] >= s[ts, n] - s[ts - 1, n])
    @constraint(opt_model, [ts in H[H.>1], n in N],
    s[ts,n] - s[ts-1,n] == charg[ts,n] - discharg[ts,n])

    #9b) Charge / Discharge (ts = 1)
    # @constraint(opt_model, [ts in [1], n in N],
    # discharg[ts, n] >= -s[ts, n] + S_0[n])
    # @constraint(opt_model, [ts in [1], n in N],
    # charg[ts, n] >= s[ts, n] - S_0[n])
    @constraint(opt_model, [ts in [1], n in N],
    s[ts,n] - S_0[n] == charg[ts,n] - discharg[ts,n])

    #10) Storage capacity
    @constraint(opt_model, [ts in H, n in N],
    s[ts,n] <= s_bar[n])

    #11) Storage continuity
    @constraint(opt_model, [n in N],
    S_0[n] == s[H[end],n])

    #12) Absolute value of transmission
    @constraint(opt_model, [ts in H, l in L],
    exp[ts,l] >= f[ts,l])
    @constraint(opt_model, [ts in H, l in L],
    exp[ts,l] >= -f[ts,l])

    # status = @time @suppress optimize!(opt_model)
    status = @time optimize!(opt_model)

    pbar_aux = value.(p_bar)
    p_aux = value.(p)
    σ_aux = value.(σ)
    fbar_aux = value.(f_bar)
    f_aux = value.(f)
    s_aux = value.(s)
    sbar_aux = value.(s_bar)
    charg_aux = value.(charg)
    discharg_aux = value.(discharg)

    sol_p_bar_int = zeros(size(pbar_aux))
    sol_p_int = zeros(size(p_aux))
    sol_σ_int = zeros(size(σ))
    sol_f_bar_int = zeros(size(fbar_aux))
    sol_f_int = zeros(size(f_aux))
    sol_s_int = zeros(size(s_aux))
    sol_s_bar_int = zeros(size(sbar_aux))
    sol_charg_int = zeros(size(charg_aux))
    sol_discharg_int = zeros(size(discharg_aux))

    for ts in H
        for n in N
            for t in T
                sol_p_int[t,ts,n] = p_aux[t,ts,n]
            end
            sol_σ_int[ts,n] = σ_aux[ts,n]
            sol_s_int[ts,n] = s_aux[ts,n]
            sol_charg_int[ts,n] = charg_aux[ts,n]
            sol_discharg_int[ts,n] = discharg_aux[ts,n]
        end
    end

    for t in T
        for n in N
            sol_p_bar_int[t,n] = pbar_aux[t,n]
        end
    end

    for ts in H
        for l in 1:length(L)
            sol_f_int[ts,l] = f_aux[ts,lines[l]]
        end
    end

    for l in 1:length(L)
        sol_f_bar_int[l] = fbar_aux[lines[l]]
    end

    for n in N
        sol_s_bar_int[n] = sbar_aux[n]
    end

    #### Breaking down of total costs:
    # K1: Total costs
    # K2: Generation Investment
    # K3: Generation Maintenance
    # K4: Generation Operational Costs
    # K5: Shedding costs
    # K6: Transmission investments
    # K7: Transmission maintenance
    # K8: Transmission Operational Costs
    # K9: Storage Investments
    # K10: Storage Operational Costs

    sol_OF_int = zeros(10)
    sol_OF_int[1] = objective_value(opt_model)
    sol_OF_int[2] = sum(K[t]*sol_p_bar_int[t,n] for t in T, n in N)
    sol_OF_int[3] = sum(KA[t]*sol_p_bar_int[t,n] for t in T, n in N)
    sol_OF_int[4] = sum(CO[t]*sol_p_int[t,ts,n] for t in T, ts in H, n in N)
    sol_OF_int[5] = sum(CS*sol_σ_int[ts,n] for ts in H,n in N)
    sol_OF_int[6] = sum(KIT[lo]*sol_f_bar_int[lo] for lo in 1:length(L))
    sol_OF_int[7] = sum(KAT*sol_f_bar_int[lo] for lo in 1:length(L))
    sol_OF_int[8] = sum(KT*abs(sol_f_int[ts,lo]) for ts in H, lo in 1:length(L))
    sol_OF_int[9] = sum(KIS*sol_s_bar_int[n] for n in N)
    sol_OF_int[10] = sum(KS*(sol_charg_int[ts,n] + sol_discharg_int[ts,n]) for ts in H, n in N)

    return sol_p_bar_int, sol_p_int, sol_σ_int, sol_OF_int,
    sol_f_bar_int , sol_f_int, sol_s_int , sol_s_bar_int,
    sol_charg_int, sol_discharg_int
end

function CTEM_clus(
    nodes::Int64,
    lines::Vector{Array{Int64,1}},
    LT::Vector{Int64},
    K::Vector{Float64},
    KA::Array{Float64},
    CO::Array{Float64},
    CS::Float64,
    GCN::Array{Float64},
    SC::Float64,
    KIT::Array{Float64},
    KT::Float64,
    T_effic::Float64,
    KIS::Float64,
    KS::Float64,
    S_effic_c::Array{Float64},
    S_effic_d::Array{Float64},
    S_0::Array{Float64},
    RES::Float64,
    τ::Int64,
    Dem_Clust_P::Array{Float64},
    Win_Clust_P::Array{Float64},
    Sol_Clust_P::Array{Float64},
    P_n_clusters_y::Union{Array{Float64},Array{Int64}},
    CS_P::Union{Array{Float64},Array{Int64}},
    max_load::Vector{Float64},
    dem_inc::Vector{Float64})

    ####   Modeling the Multi-nodal CTEM for 5 different technologies (clus)  ###
    ## Data to be used:
    # lines: possible lines connections
    # nodes: Number of nodes
    # τ: Number of time slices per month
    # Dem_Clust_P: Demand (clusterred by Pineda's)
    # Win_Clust_P: Wind availability (clusterred by Pineda's)
    # Sol_Clust_P: Solar availability (clusterred by Pineda's)
    # RES: level of renewables needed into the system
    # CS_P: Clusters sizes (672,8064)
    # max_load: Max load of each node
    # dem_inc: Demand increase projection

    #______________________________________________________________________________#
    ############################    Sets    ########################################
    #______________________________________________________________________________#

    TS = collect(1:Int.(P_n_clusters_y[τ]))     # Set of time steps (Reduced approach)
    T = collect(1:5)                    # Technologies
    N = collect(1:nodes)                # Nodes set
    L = copy(lines)                     # Lines set

    ## Technologies
    #   1: Wind
    #   2: Solar
    #   3: Coal
    #   4: Gas CC
    #   5: Gas OC
    ##

    # #______________________________________________________________________________#
    # ############################    Parameters    ##################################
    # #______________________________________________________________________________#
    #
    Dem = zeros(TS[end], N[end])
    Win = zeros(TS[end], N[end])
    Sol = zeros(TS[end], N[end])

    ## Full-time resolution Demand for 2030
    for n in N
        Dem[:,n] = dem_inc[n].*Dem_Clust_P[1:TS[end],n].*max_load[n]             # Integral Demand in each node (MWh)
        Win[:,n] = Win_Clust_P[1:TS[end],n]             # Integral Demand in each node (MWh)
        Sol[:,n] = Sol_Clust_P[1:TS[end],n]             # Integral Demand in each node (MWh)
    end

    ############################################################################
    ###################    MULTI-NODAL MODEL (integral)    #####################
    ############################################################################

    #__________________________________________________________________________#
    ###########################    Variables    ################################
    #__________________________________________________________________________#
    # opt_model = Model(Gurobi.Optimizer)
    opt_model = Model(Gurobi.Optimizer)

    @variable(opt_model, p[t in T, ts in TS, n in N]>=0)    # Generation level in each hour by t [MW]
    @variable(opt_model, p_bar[t in T, n in N]>=0)          # Additional generation capac. of t [MW]
    @variable(opt_model, σ[ts in TS, n in N]>=0)            # Energy shedding in ts [MWh]
    @variable(opt_model, f[ts in TS, l in L])               # Transmission flow in each hour and line [MWh]
    @variable(opt_model, exp[ts in TS, l in L]>=0)          # Transmission flow in absolute [MWh]
    @variable(opt_model, f_bar[l in L]>=0)                  # Additional transmission capac. of t [MW]
    @variable(opt_model, s[ts in TS, n in N] >= 0)          # Storage level in each hour [MWh]
    @variable(opt_model, s_bar[n in N]>=0)                  # Storage capac. of t [MWh]
    @variable(opt_model, charg[ts in TS, n in N] >= 0)      # Charging in each hour [MWh]
    @variable(opt_model, discharg[ts in TS, n in N] >= 0)   # Discharging in each hour [MWh]

    #______________________________________________________________________________#
    ########################    Objective Function    ##############################
    #______________________________________________________________________________#

    @objective(opt_model, Min, sum((K[t]+KA[t])*p_bar[t,n] for t in T, n in N) +
                               sum(CO[t]*p[t,ts,n]*CS_P[ts] for t in T, ts in TS, n in N) +
                               sum(CS*σ[ts,n]*CS_P[ts] for ts in TS,n in N) +
                               sum((KIT[lo] + KAT)*f_bar[L[lo]] for lo in 1:length(L)) +
                               sum(KT*exp[ts,l]*CS_P[ts] for ts in TS, l in L) +
                               sum(KIS*s_bar[n] for n in N) +
                               sum(KS*(charg[ts,n] + discharg[ts,n]).*CS_P[ts] for ts in TS, n in N))
    #
    #______________________________________________________________________________#
    ############################    Constraints    #################################
    #______________________________________________________________________________#

    #1a) Energy balance (ts > 1)
    @constraint(opt_model,[ts in TS[TS.>1],n in N],
    sum(p[t,ts,n] for t in T) + σ[ts,n] -
    sum(f[ts,l] for l in L if l[1] .== n) +
    sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    S_effic_c[n]*(s[ts, n] - s[ts - 1, n])
    == Dem[ts,n])

    #1b) Energy balance (ts = 1)
    @constraint(opt_model,[ts in [1],n in N],
    sum(p[t,ts,n] for t in T) + σ[ts,n] -
    sum(f[ts,l] for l in L if l[1] .== n) +
    sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    S_effic_c[n]*(s[ts, n] - S_0[n])
    == Dem[ts,n])

    #2) Wind generation capacity
    @constraint(opt_model,[t in [1],ts in TS,n in N],
    p[t,ts,n] <= Win[ts,n].*(GCN[t,n] + p_bar[t,n]))

    #3) Solar generation capacity
    @constraint(opt_model,[t in [2],ts in TS,n in N],
    p[t,ts,n] <= Sol[ts,n].*(GCN[t,n] + p_bar[t,n]))

    #4) NRES capacity
    @constraint(opt_model,[t in [3,4,5],ts in TS,n in N],
    p[t,ts,n] <= (GCN[t,n] + p_bar[t,n]))

    #5) Minimum RES
    @constraint(opt_model,
    sum(p[t,ts,n] for t in [1,2],ts in TS,n in N) >=
    RES*sum(p[t,ts,n] for t in T,ts in TS,n in N))

    # @constraint(opt_model,
    # sum(p[t,ts,n] for t in [1,2],ts in TS,n in N) >=
    # sum(RES3*Dem[ts,n] for ts in TS, n in N))

    # #6) Budget?
    # @constraint(opt_model, sum((K[t]+KA[t]).*p_bar[t] for t in T, ts in TS) <= B)

    #7) Shedding cap
    @constraint(opt_model,[ts in TS, n in N],
    σ[ts,n] <= SC*Dem[ts,n])

    #8) Transmission cap
    @constraint(opt_model,[ts in TS, l in L],
    f[ts,l] <= f_bar[l])
    @constraint(opt_model,[ts in TS, l in L],
    -f_bar[l] <= f[ts,l])

    #9a) Charge / Discharge (ts > 1)
    @constraint(opt_model, [ts in TS[TS.>1], n in N],
    s[ts,n] - s[ts-1,n] == charg[ts,n] - discharg[ts,n])

    #9b) Charge / Discharge (ts = 1)
    @constraint(opt_model, [ts in [1], n in N],
    s[ts,n] - S_0[n] == charg[ts,n] - discharg[ts,n])

    #10) Storage capacity
    @constraint(opt_model, [ts in TS, n in N],
    s[ts,n] <= s_bar[n])

    #11) Storage continuity
    @constraint(opt_model, [n in N],
    S_0[n] == s[TS[end],n])

    #12) Absolute value of transmission
    @constraint(opt_model, [ts in TS, l in L],
    exp[ts,l] >= f[ts,l])
    @constraint(opt_model, [ts in TS, l in L],
    exp[ts,l] >= -f[ts,l])

    # status = @time @suppress optimize!(opt_model)
    status = @time optimize!(opt_model)

    pbar_aux = value.(p_bar)
    p_aux = value.(p)
    σ_aux = value.(σ)
    fbar_aux = value.(f_bar)
    f_aux = value.(f)
    s_aux = value.(s)
    sbar_aux = value.(s_bar)
    charg_aux = value.(charg)
    discharg_aux = value.(discharg)

    sol_p_bar_clus = zeros(size(pbar_aux))
    sol_p_clus = zeros(size(p_aux))
    sol_σ_clus = zeros(size(σ))
    sol_f_bar_clus = zeros(size(fbar_aux))
    sol_f_clus = zeros(size(f_aux))
    sol_s_clus = zeros(size(s_aux))
    sol_s_bar_clus = zeros(size(sbar_aux))
    sol_charg_clus = zeros(size(charg_aux))
    sol_discharg_clus = zeros(size(discharg_aux))

    for ts in TS
        for n in N
            for t in T
                sol_p_clus[t,ts,n] = p_aux[t,ts,n]
            end
            sol_σ_clus[ts,n] = σ_aux[ts,n]
            sol_s_clus[ts,n] = s_aux[ts,n]
            sol_charg_clus[ts,n] = charg_aux[ts,n]
            sol_discharg_clus[ts,n] = discharg_aux[ts,n]
        end
    end

    for t in T
        for n in N
            sol_p_bar_clus[t,n] = pbar_aux[t,n]
        end
    end

    for ts in TS
        for l in 1:length(L)
            sol_f_clus[ts,l] = f_aux[ts,lines[l]]
        end
    end

    for l in 1:length(L)
        sol_f_bar_clus[l] = fbar_aux[lines[l]]
    end

    for n in N
        sol_s_bar_clus[n] = sbar_aux[n]
    end

    #### Breaking down of total costs:
    # K1: Total costs
    # K2: Generation Investment
    # K3: Generation Maintenance
    # K4: Generation Operational Costs
    # K5: Shedding costs
    # K6: Transmission investments
    # K7: Transmission maintenance
    # K8: Transmission Operational Costs
    # K9: Storage Investments
    # K10: Storage Operational Costs

    sol_OF_clus = zeros(10)
    sol_OF_clus[1] = objective_value(opt_model)
    sol_OF_clus[2] = sum(K[t]*sol_p_bar_clus[t,n] for t in T, n in N)
    sol_OF_clus[3] = sum(KA[t]*sol_p_bar_clus[t,n] for t in T, n in N)
    sol_OF_clus[4] = sum(CO[t]*sol_p_clus[t,ts,n]*CS_P[ts] for t in T, ts in TS, n in N)
    sol_OF_clus[5] = sum(CS*sol_σ_clus[ts,n]*CS_P[ts] for ts in TS,n in N)
    sol_OF_clus[6] = sum(KIT[lo]*sol_f_bar_clus[lo] for lo in 1:length(L))
    sol_OF_clus[7] = sum(KAT*sol_f_bar_clus[lo] for lo in 1:length(L))
    sol_OF_clus[8] = sum(KT*abs(sol_f_clus[ts,lo]) for ts in TS, lo in 1:length(L))
    sol_OF_clus[9] = sum(KIS*sol_s_bar_clus[n] for n in N)
    sol_OF_clus[10] = sum(KS*(sol_charg_clus[ts,n] + sol_discharg_clus[ts,n])*CS_P[ts] for ts in TS, n in N)

    return sol_p_bar_clus, sol_p_clus, sol_σ_clus, sol_OF_clus,
    sol_f_bar_clus , sol_f_clus, sol_s_bar_clus , sol_s_clus,
    sol_charg_clus, sol_discharg_clus
end

function compute_RMSE(
    is_nodal::Bool,
    Cluss::UnitRange{Int64},
    s_clus_Gen_Inv::Array{Float64},
    s_int_Gen_Inv::Array{Float64},
    s_clus_Trans_Inv::Array{Float64},
    s_int_Trans_Inv::Array{Float64},
    s_clus_Stor_Inv::Array{Float64},
    s_int_Stor_Inv::Array{Float64}
    )

    RMSE_P = zeros(length(Cluss))

    if is_nodal
        for l in Cluss
            time_aux = now()
            # println("l is $l in $time_aux")
            RMSE_P[l] =
            sum(sqrt.(sum((s_clus_Gen_Inv[l,t] .- s_int_Gen_Inv[t]).^2 for t in 1:5))) +
            sum(sqrt.(sum((s_clus_Stor_Inv[l] .- s_int_Stor_Inv).^2)))
        end
    else
        for l in Cluss
            time_aux = now()
            # println("l is $l in $time_aux")
            RMSE_P[l] =
            sum(sqrt.(sum((s_clus_Gen_Inv[l,t,:] .- s_int_Gen_Inv[t,:]).^2 for t in 1:5))) +
            sum(sqrt.(sum((s_clus_Trans_Inv[l,:] .- s_int_Trans_Inv[:]).^2))) +
            sum(sqrt.(sum((s_clus_Stor_Inv[l,:] .- s_int_Stor_Inv[:]).^2)))
        end
    end
    return RMSE_P
end

################################################################################
### NOTE: Nodal functions
################################################################################

function compute_RMSE_inv_Nodal(
    Cluss::UnitRange{Int64},
    s_clus_Gen_Inv::Array{Float64},
    s_int_Gen_Inv::Array{Float64},
    s_clus_Stor_Inv::Array{Float64},
    s_int_Stor_Inv::Array{Float64}
    )

    RMSE_P = zeros(length(Cluss))

    for l in Cluss
        time_aux = now()
        # println("l is $l in $time_aux")
        RMSE_P[l] =
        sum(sqrt.(sum((s_clus_Gen_Inv[l,t] .- s_int_Gen_Inv[t]).^2 for t in 1:5))) +
        sum(sqrt.(sum((s_clus_Stor_Inv[l] .- s_int_Stor_Inv).^2 for t in 1:5)))
    end

    return RMSE_P
end

function CEM_int(
    # nodes::Int64,
    # lines::Vector{Array{Int64,1}},
    # LT::Vector{Int64},
    K::Vector{Float64},
    KA::Vector{Float64},
    CO::Vector{Float64},
    CS::Float64,
    GCN::Vector{Float64},
    SC::Float64,
    # KIT::Array{Float64},
    # KT::Float64,
    # T_effic::Float64,
    KIS::Float64,
    KS::Float64,
    S_effic::Float64,
    RES::Float64,
    D::Vector{Float64},
    W_avail::Vector{Float64},
    S_avail::Vector{Float64},
    max_load::Float64,
    dem_inc::Float64
    )

    ####   Modeling the Nodal CEM for 5 different technologies (int)  ###
    ## Data to be used:
    # RES: level of renewables needed into the system
    # nodes: Number of nodes
    # D: Demand
    # W_avail: Wind availability
    # S_avail: Solar availability

    #______________________________________________________________________________#
    ############################    Sets    ########################################
    #______________________________________________________________________________#

    H = collect(1:length(D))                 # Total hours (Integral approach)
    T = collect(1:5)                    # Technologies
    # N = collect(1:nodes)                # Nodes set
    # L = copy(lines)                     # Lines set

    ## Technologies
    #   1: Wind
    #   2: Solar
    #   3: Coal
    #   4: Gas
    #   5: Oil
    ##

    #______________________________________________________________________________#
    ############################    Parameters    ##################################
    #______________________________________________________________________________#

    Dem = zeros(H[end])

    ## Full-time resolution Demand for 2030
    Dem = dem_inc.*D.*max_load             # Naively forecasted Demand (MWh)


    ############################################################################
    ###################    MULTI-NODAL MODEL (integral)    #####################
    ############################################################################

    #__________________________________________________________________________#
    ###########################    Variables    ################################
    #__________________________________________________________________________#
    opt_model = Model(Gurobi.Optimizer)

    @variable(opt_model, p[t in T, ts in H]>=0)    # Generation level in each hour by t [MW]
    @variable(opt_model, p_bar[t in T]>=0)         # Additional generation capac. of t [MW]
    @variable(opt_model, σ[ts in H]>=0)            # Energy shedding in ts [MWh]
    # @variable(opt_model, f[ts in H, l in L])               # Transmission flow in each hour and line [MWh]
    # @variable(opt_model, exp[ts in H, l in L]>=0)             # Positive transmission flow in each and line [MWh]
    # @variable(opt_model, f_bar[l in L]>=0)                 # Additional transmission capac. of t [MW]
    @variable(opt_model, s[ts in H] >= 0)          # Storage level in each hour [MWh]
    @variable(opt_model, s_bar>=0)                 # Storage capac. of t [MWh]
    @variable(opt_model, charg[ts in H] >= 0)      # Charging in each hour [MWh]
    @variable(opt_model, discharg[ts in H] >= 0)   # Discharging in each hour [MWh]

    #______________________________________________________________________________#
    ########################    Objective Function    ##############################
    #______________________________________________________________________________#

    @objective(opt_model, Min, sum((K[t]+KA[t])*p_bar[t] for t in T) +
                               sum(CO[t]*p[t,ts] for t in T, ts in H) +
                               sum(CS*σ[ts] for ts in H) +
                               # sum((KIT[lo] + KAT)*f_bar[L[lo]] for lo in 1:length(L)) +
                               # sum(KT*exp[ts,l] for ts in H, l in L) +
                               sum(KIS*s_bar) +
                               sum(KS*(charg[ts] + discharg[ts]) for ts in H)
    )
    #
    #______________________________________________________________________________#
    ############################    Constraints    #################################
    #______________________________________________________________________________#

    #1a) Energy balance (ts > 1)
    @constraint(opt_model,[ts in H[H.>1]],
    sum(p[t,ts] for t in T) + σ[ts] -
    S_effic*(s[ts] - s[ts - 1])
    == Dem[ts])

    #1b) Energy balance (ts = 1)
    @constraint(opt_model,[ts in [1]],
    sum(p[t,ts] for t in T) + σ[ts] -
    S_effic*(s[ts] - S0[1])
    == Dem[ts])

    #2) Wind generation capacity
    @constraint(opt_model,[t in [1],ts in H],
    p[t,ts] <= W_avail[ts].*(GCN[t] + p_bar[t]))

    #3) Solar generation capacity
    @constraint(opt_model,[t in [2],ts in H],
    p[t,ts] <= S_avail[ts].*(GCN[t] + p_bar[t]))

    #4) NRES capacity
    @constraint(opt_model,[t in [3,4,5],ts in H],
    p[t,ts] <= (GCN[t] + p_bar[t]))

    #5) Minimum RES
    @constraint(opt_model,
    sum(p[t,ts] for t in [1,2],ts in H) >=
    RES*sum(p[t,ts] for t in T,ts in H))

    # @constraint(opt_model,
    # sum(p[t,ts,n] for t in [1,2],ts in H,n in N) >=
    # sum(RES3*Dem[ts,n] for ts in H, n in N))

    # #6) Budget?
    # @constraint(opt_model, sum((K[t]+KA[t]).*p_bar[t] for t in T, ts in TS) <= B)

    #7) Shedding cap
    @constraint(opt_model,[ts in H],
    σ[ts] <= SC*Dem[ts])

    # #8) Transmission cap
    # @constraint(opt_model,[ts in H, l in L],
    # f[ts,l] <= f_bar[l])
    # @constraint(opt_model,[ts in H, l in L],
    # -f_bar[l] <= f[ts,l])

    #9a) Charge / Discharge (ts > 1)
    @constraint(opt_model, [ts in H[H.>1]],
    discharg[ts] >= -s[ts] + s[ts - 1])
    @constraint(opt_model, [ts in H[H.>1]],
    charg[ts] >= s[ts] - s[ts - 1])

    #9b) Charge / Discharge (ts = 1)
    @constraint(opt_model, [ts in [1]],
    discharg[ts] >= -s[ts] + S0[1])
    @constraint(opt_model, [ts in [1]],
    charg[ts] >= s[ts] - S0[1])

    #10) Storage capacity
    @constraint(opt_model, [ts in H],
    s[ts] <= s_bar)

    #11) Storage continuity
    @constraint(opt_model,
    s[1] == s[T[end]])

    # #12) Absolute value of transmission
    # @constraint(opt_model, [ts in H, l in L],
    # exp[ts,l] >= f[ts,l])
    # @constraint(opt_model, [ts in H, l in L],
    # exp[ts,l] >= -f[ts,l])

    # status = @time @suppress optimize!(opt_model)
    status = @time optimize!(opt_model)

    pbar_aux = value.(p_bar)
    p_aux = value.(p)
    σ_aux = value.(σ)
    # fbar_aux = value.(f_bar)
    # f_aux = value.(f)
    s_aux = value.(s)
    sbar_aux = value.(s_bar)

    sol_p_bar_int = zeros(size(pbar_aux))
    sol_p_int = zeros(size(p_aux))
    sol_σ_int = zeros(size(σ))
    # sol_f_bar_int = zeros(size(fbar_aux))
    # sol_f_int = zeros(size(f_aux))
    sol_s_int = zeros(size(s_aux))
    sol_s_bar_int = zeros(size(sbar_aux))

    for ts in H
        for t in T
            sol_p_int[t,ts] = p_aux[t,ts]
        end
        sol_σ_int[ts] = σ_aux[ts]
        sol_s_int[ts] = s_aux[ts]
    end

    for t in T
        sol_p_bar_int[t] = pbar_aux[t]
    end

    # for ts in H
    #     for l in 1:length(L)
    #         sol_f_int[ts,l] = f_aux[ts,lines[l]]
    #     end
    # end
    #
    # for l in 1:length(L)
    #     sol_f_bar_int[l] = fbar_aux[lines[l]]
    # end

    sol_s_bar_int = sbar_aux

    sol_OF_int = objective_value(opt_model)

    return sol_p_bar_int, sol_p_int, sol_σ_int, sol_OF_int,
    sol_s_int , sol_s_bar_int
end

function CEM_clus(
    K::Vector{Float64},
    KA::Vector{Float64},
    CO::Vector{Float64},
    CS::Float64,
    GCN::Vector{Float64},
    SC::Float64,
    KIS::Float64,
    KS::Float64,
    S_effic::Float64,
    RES::Float64,
    τ::Int64,
    Dem_Clust_P::Matrix{Float64},
    Win_Clust_P::Matrix{Float64},
    Sol_Clust_P::Matrix{Float64},
    P_n_clusters_y::Vector{Float64},
    CS_P::Vector{Float64},
    max_load::Vector{Float64},
    dem_inc::Vector{Float64})


    ####   Modeling the Multi-nodal CTEM for 5 different technologies (clus)  ###
    ## Data to be used:
    # lines: possible lines connections
    # nodes: Number of nodes
    # τ: Number of time slices per month
    # Dem_Clust_P: Demand (clusterred by Pineda's)
    # Win_Clust_P: Wind availability (clusterred by Pineda's)
    # Sol_Clust_P: Solar availability (clusterred by Pineda's)
    # RES: level of renewables needed into the system
    # CS_P: Clusters sizes (672,8064)
    # max_load: Max load of each node
    # dem_inc: Demand increase projection

    #______________________________________________________________________________#
    ############################    Sets    ########################################
    #______________________________________________________________________________#

    TS = collect(1:Int.(P_n_clusters_y[τ]))     # Set of time steps (Reduced approach)
    T = collect(1:5)                    # Technologies

    ## Technologies
    #   1: Wind
    #   2: Solar
    #   3: Coal
    #   4: Gas CC
    #   5: Gas OC
    ##

    # #______________________________________________________________________________#
    # ############################    Parameters    ##################################
    # #______________________________________________________________________________#
    #
    Dem = zeros(TS[end])
    Win = zeros(TS[end])
    Sol = zeros(TS[end])

    ## Full-time resolution Demand for 2030
    Dem = dem_inc.*Dem_Clust_P[1:TS[end]].*max_load             # Integral Demand in each node (MWh)
    Win = Win_Clust_P[1:TS[end]]             # Integral Demand in each node (MWh)
    Sol = Sol_Clust_P[1:TS[end]]             # Integral Demand in each node (MWh)

    ############################################################################
    ###################    MULTI-NODAL MODEL (integral)    #####################
    ############################################################################

    #__________________________________________________________________________#
    ###########################    Variables    ################################
    #__________________________________________________________________________#
    opt_model = Model(Gurobi.Optimizer)

    @variable(opt_model, p[t in T, ts in TS]>=0)    # Generation level in each hour by t [MW]
    @variable(opt_model, p_bar[t in T]>=0)         # Additional generation capac. of t [MW]
    @variable(opt_model, σ[ts in TS]>=0)            # Energy shedding in ts [MW]
    @variable(opt_model, s[ts in TS] >= 0)          # Storage level in each hour [MWh]
    @variable(opt_model, s_bar>=0)                 # Storage capac. of t [MWh]
    @variable(opt_model, charg[ts in TS] >= 0)      # Charging in each hour [MWh]
    @variable(opt_model, discharg[ts in TS] >= 0)   # Discharging in each hour [MWh]

    #______________________________________________________________________________#
    ########################    Objective Function    ##############################
    #______________________________________________________________________________#

    @objective(opt_model, Min, sum((K[t]+KA[t]).*p_bar[t] for t in T) +
                               sum(CO[t].*p[t,ts].*CS_P[ts] for t in T, ts in TS) +
                               sum(CS*σ[ts].*CS_P[ts] for ts in TS) +
                               KIS*s_bar +
                               sum(KS*(charg[ts] + discharg[ts]).*CS_P[ts] for ts in TS))
    #
    #______________________________________________________________________________#
    ############################    Constraints    #################################
    #______________________________________________________________________________#

    #1a) Energy balance (ts > 1)
    @constraint(opt_model,[ts in TS[TS.>1]],
    sum(p[t,ts] for t in T) + σ[ts] -
    S_effic*(s[ts] - s[ts - 1])
    == Dem[ts])

    #1b) Energy balance (ts = 1)
    @constraint(opt_model,[ts in [1]],
    sum(p[t,ts] for t in T) + σ[ts] -
    S_effic*(s[ts])
    == Dem[ts])

    #2) Wind generation capacity
    @constraint(opt_model,[t in [1],ts in TS],
    p[t,ts] <= Win[ts].*(GCN[t] + p_bar[t]))

    #3) Solar generation capacity
    @constraint(opt_model,[t in [2],ts in TS],
    p[t,ts] <= Sol[ts].*(GCN[t] + p_bar[t]))

    #4) NRES capacity
    @constraint(opt_model,[t in [3,4,5],ts in TS],
    p[t,ts] <= (GCN[t] + p_bar[t]))

    #5) Minimum RES
    @constraint(opt_model,
    sum(p[t,ts] for t in [1,2],ts in TS) >=
    RES*sum(p[t,ts] for t in T,ts in TS))

    # @constraint(opt_model,
    # sum(p[t,ts,n] for t in [1,2],ts in TS,n in N) >=
    # sum(RES3*Dem[ts,n] for ts in TS, n in N))

    # #6) Budget?
    # @constraint(opt_model, sum((K[t]+KA[t]).*p_bar[t] for t in T, ts in TS) <= B)

    #7) Shedding cap
    @constraint(opt_model,[ts in TS],
    σ[ts] <= SC*Dem[ts])

    #9a) Charge / Discharge (ts > 1)
    @constraint(opt_model, [ts in TS[TS.>1]],
    discharg[ts] >= -s[ts] + s[ts - 1])
    @constraint(opt_model, [ts in TS[TS.>1]],
    charg[ts] >= s[ts] - s[ts - 1])

    #9b) Charge / Discharge (ts = 1)
    @constraint(opt_model, [ts in [1]],
    discharg[ts] >= s[ts] - S0[1])
    @constraint(opt_model, [ts in [1]],
    charg[ts] >= s[ts] - S0[1])

    #10) Storage capacity
    @constraint(opt_model, [ts in TS],
    s[ts]*CS_P[ts] <= s_bar)

    #11) Storage continuity
    @constraint(opt_model,
    s[1] == s[T[end]])


    # status = @time @suppress optimize!(opt_model)
    status = @time optimize!(opt_model)

    pbar_aux = value.(p_bar)
    p_aux = value.(p)
    σ_aux = value.(σ)
    s_aux = value.(s)
    sbar_aux = value.(s_bar)

    sol_p_bar_clus = zeros(size(pbar_aux))
    sol_p_clus = zeros(size(p_aux))
    sol_σ_clus = zeros(size(σ))
    sol_s_clus = zeros(size(s_aux))
    sol_s_bar_clus = zeros(size(sbar_aux))

    for ts in TS
        for t in T
            sol_p_clus[t,ts] = p_aux[t,ts]
        end
        sol_σ_clus[ts] = σ_aux[ts]
        sol_s_clus[ts] = s_aux[ts]
    end

    for t in T
        sol_p_bar_clus[t] = pbar_aux[t]
    end

    sol_s_bar_clus = sbar_aux

    sol_OF_clus = objective_value(opt_model)

    return sol_p_bar_clus, sol_p_clus, sol_σ_clus, sol_OF_clus,
    sol_s_bar_clus , sol_s_clus
end

################################################################################
### NOTE: Fixed capacity
################################################################################

function fix_cap(
    nodes::Int64,
    lines::Vector{Array{Int64,1}},
    LT::Vector{Int64},
    K::Vector{Float64},
    KA::Array{Float64},
    CO::Array{Float64},
    CS::Float64,
    GCN::Array{Float64},
    SC::Float64,
    KIT::Array{Float64},
    KT::Float64,
    T_effic::Float64,
    KIS::Float64,
    KS::Float64,
    S_effic_c::Array{Float64},
    S_effic_d::Array{Float64},
    S_0::Array{Float64},
    RES::Float64,
    τ::Int64,
    Dem_Clust_P::Array{Float64},
    Win_Clust_P::Array{Float64},
    Sol_Clust_P::Array{Float64},
    P_n_clusters_y::Array{Float64},
    CS_P::Array{Float64},
    max_load::Vector{Float64},
    dem_inc::Vector{Float64},
    s_int_Gen_Inv::Array{Float64},
    s_int_Trans_Inv::Array{Float64},
    s_int_Stor_Inv::Array{Float64}
    )

    ####   Modeling the Multi-nodal CTEM for 5 different technologies (clus)  ###
    ## Data to be used:
    # lines: possible lines connections
    # nodes: Number of nodes
    # τ: Number of time slices per month
    # Dem_Clust_P: Demand (clusterred by Pineda's)
    # Win_Clust_P: Wind availability (clusterred by Pineda's)
    # Sol_Clust_P: Solar availability (clusterred by Pineda's)
    # RES: level of renewables needed into the system
    # CS_P: Clusters sizes (672,8064)
    # max_load: Max load of each node
    # dem_inc: Demand increase projection

    #______________________________________________________________________________#
    ############################    Sets    ########################################
    #______________________________________________________________________________#

    TS = collect(1:Int.(P_n_clusters_y[τ]))     # Set of time steps (Reduced approach)
    T = collect(1:5)                    # Technologies
    N = collect(1:nodes)                # Nodes set
    L = copy(lines)                     # Lines set

    ## Technologies
    #   1: Wind
    #   2: Solar
    #   3: Coal
    #   4: Gas CC
    #   5: Gas OC
    ##

    # #______________________________________________________________________________#
    # ############################    Parameters    ##################################
    # #______________________________________________________________________________#
    #
    Dem = zeros(TS[end], N[end])
    Win = zeros(TS[end], N[end])
    Sol = zeros(TS[end], N[end])

    ## Full-time resolution Demand for 2030
    for n in N
        Dem[:,n] = dem_inc[n].*Dem_Clust_P[1:TS[end],n].*max_load[n]             # Integral Demand in each node (MWh)
        Win[:,n] = Win_Clust_P[1:TS[end],n]             # Integral Demand in each node (MWh)
        Sol[:,n] = Sol_Clust_P[1:TS[end],n]             # Integral Demand in each node (MWh)
    end

    ############################################################################
    ###################    MULTI-NODAL MODEL (integral)    #####################
    ############################################################################

    #__________________________________________________________________________#
    ###########################    Variables    ################################
    #__________________________________________________________________________#
    # opt_model = Model(Gurobi.Optimizer)
    opt_model = Model(Gurobi.Optimizer)

    @variable(opt_model, p[t in T, ts in TS, n in N]>=0)    # Generation level in each hour by t [MW]
    @variable(opt_model, p_bar[t in T, n in N]>=0)          # Additional generation capac. of t [MW]
    @variable(opt_model, σ[ts in TS, n in N]>=0)            # Energy shedding in ts [MWh]
    @variable(opt_model, f[ts in TS, l in L])               # Transmission flow in each hour and line [MWh]
    @variable(opt_model, exp[ts in TS, l in L])             # Transmission flow in absolute [MWh]
    @variable(opt_model, f_bar[l in L]>=0)                  # Additional transmission capac. of t [MW]
    @variable(opt_model, s[ts in TS, n in N] >= 0)          # Storage level in each hour [MWh]
    @variable(opt_model, s_bar[n in N]>=0)                  # Storage capac. of t [MWh]
    @variable(opt_model, charg[ts in TS, n in N] >= 0)      # Charging in each hour [MWh]
    @variable(opt_model, discharg[ts in TS, n in N] >= 0)   # Discharging in each hour [MWh]

    #______________________________________________________________________________#
    ########################    Objective Function    ##############################
    #______________________________________________________________________________#

    @objective(opt_model, Min, sum((K[t]+KA[t])*p_bar[t,n] for t in T, n in N) +
                               sum(CO[t]*p[t,ts,n]*CS_P[ts] for t in T, ts in TS, n in N) +
                               sum(CS*σ[ts,n]*CS_P[ts] for ts in TS,n in N) +
                               sum((KIT[lo] + KAT)*f_bar[L[lo]] for lo in 1:length(L)) +
                               sum(KT*exp[ts,l]*CS_P[ts] for ts in TS, l in L) +
                               sum(KIS*s_bar[n] for n in N) +
                               sum(KS*(charg[ts,n] + discharg[ts,n]).*CS_P[ts] for ts in TS, n in N))
    #
    #______________________________________________________________________________#
    ############################    Constraints    #################################
    #______________________________________________________________________________#

    #1a) Energy balance (ts > 1)
    @constraint(opt_model,[ts in TS[TS.>1],n in N],
    sum(p[t,ts,n] for t in T) + σ[ts,n] -
    sum(f[ts,l] for l in L if l[1] .== n) +
    sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    S_effic*(s[ts, n] - s[ts - 1, n])
    == Dem[ts,n])

    #1b) Energy balance (ts = 1)
    @constraint(opt_model,[ts in [1],n in N],
    sum(p[t,ts,n] for t in T) + σ[ts,n] -
    sum(f[ts,l] for l in L if l[1] .== n) +
    sum(T_effic*f[ts,l] for l in L if l[2] .== n) -
    S_effic*(s[ts, n])
    == Dem[ts,n])

    #2) Wind generation capacity
    @constraint(opt_model,[t in [1],ts in TS,n in N],
    p[t,ts,n] <= Win[ts,n].*(GCN[t,n] + p_bar[t,n]))

    #3) Solar generation capacity
    @constraint(opt_model,[t in [2],ts in TS,n in N],
    p[t,ts,n] <= Sol[ts,n].*(GCN[t,n] + p_bar[t,n]))

    #4) NRES capacity
    @constraint(opt_model,[t in [3,4,5],ts in TS,n in N],
    p[t,ts,n] <= (GCN[t,n] + p_bar[t,n]))

    #5) Minimum RES
    @constraint(opt_model,
    sum(p[t,ts,n] for t in [1,2],ts in TS,n in N) >=
    RES*sum(p[t,ts,n] for t in T,ts in TS,n in N))

    # @constraint(opt_model,
    # sum(p[t,ts,n] for t in [1,2],ts in TS,n in N) >=
    # sum(RES3*Dem[ts,n] for ts in TS, n in N))

    # #6) Budget?
    # @constraint(opt_model, sum((K[t]+KA[t]).*p_bar[t] for t in T, ts in TS) <= B)

    #7) Shedding cap
    @constraint(opt_model,[ts in TS, n in N],
    σ[ts,n] <= SC*Dem[ts,n])

    #8) Transmission cap
    @constraint(opt_model,[ts in TS, l in L],
    f[ts,l] <= f_bar[l])
    @constraint(opt_model,[ts in TS, l in L],
    -f_bar[l] <= f[ts,l])

    #9a) Charge / Discharge (ts > 1)
    @constraint(opt_model, [ts in TS[TS.>1], n in N],
    s[ts,n] - s[ts-1,n] == charg[ts,n] - discharg[ts,n])

    #9b) Charge / Discharge (ts = 1)
    @constraint(opt_model, [ts in [1], n in N],
    s[ts,n] - S_0[n] == charg[ts,n] - discharg[ts,n])

    #10) Storage capacity
    @constraint(opt_model, [ts in TS, n in N],
    s[ts,n] <= s_bar[n])

    #11) Storage continuity
    @constraint(opt_model, [n in N],
    S_0[n] == s[TS[end],n])

    #12) Absolute value of transmission
    @constraint(opt_model, [ts in TS, l in L],
    exp[ts,l] >= f[ts,l])
    @constraint(opt_model, [ts in TS, l in L],
    exp[ts,l] >= -f[ts,l])

    #13) Constant capacity
    @constraint(opt_model, [n in N, t in T],
    p_bar[t,n] >= s_int_Gen_Inv[t,n])
    @constraint(opt_model, [lo in 1:length(L)],
    f_bar[L[lo]] >= s_int_Trans_Inv[lo])
    @constraint(opt_model, [n in N],
    s_bar[n] >= s_int_Stor_Inv[n])

    # status = @time @suppress optimize!(opt_model)
    status = @time optimize!(opt_model)

    pbar_aux = value.(p_bar)
    p_aux = value.(p)
    σ_aux = value.(σ)
    fbar_aux = value.(f_bar)
    f_aux = value.(f)
    s_aux = value.(s)
    sbar_aux = value.(s_bar)
    charg_aux = value.(charg)
    discharg_aux = value.(discharg)

    sol_p_bar_clus = zeros(size(pbar_aux))
    sol_p_clus = zeros(size(p_aux))
    sol_σ_clus = zeros(size(σ))
    sol_f_bar_clus = zeros(size(fbar_aux))
    sol_f_clus = zeros(size(f_aux))
    sol_s_clus = zeros(size(s_aux))
    sol_s_bar_clus = zeros(size(sbar_aux))
    sol_charg_clus = zeros(size(charg_aux))
    sol_discharg_clus = zeros(size(discharg_aux))

    for ts in TS
        for n in N
            for t in T
                sol_p_clus[t,ts,n] = p_aux[t,ts,n]
            end
            sol_σ_clus[ts,n] = σ_aux[ts,n]
            sol_s_clus[ts,n] = s_aux[ts,n]
            sol_charg_clus[ts,n] = charg_aux[ts,n]
            sol_discharg_clus[ts,n] = discharg_aux[ts,n]
        end
    end

    for t in T
        for n in N
            sol_p_bar_clus[t,n] = pbar_aux[t,n]
        end
    end

    for ts in TS
        for l in 1:length(L)
            sol_f_clus[ts,l] = f_aux[ts,lines[l]]
        end
    end

    for l in 1:length(L)
        sol_f_bar_clus[l] = fbar_aux[lines[l]]
    end

    for n in N
        sol_s_bar_clus[n] = sbar_aux[n]
    end

    #### Breaking down of total costs:
    # K1: Total costs
    # K2: Generation Investment
    # K3: Generation Maintenance
    # K4: Generation Operational Costs
    # K5: Shedding costs
    # K6: Transmission investments
    # K7: Transmission maintenance
    # K8: Transmission Operational Costs
    # K9: Storage Investments
    # K10: Storage Operational Costs

    sol_OF_clus = zeros(10)
    sol_OF_clus[1] = objective_value(opt_model)
    sol_OF_clus[2] = sum(K[t]*sol_p_bar_clus[t,n] for t in T, n in N)
    sol_OF_clus[3] = sum(KA[t]*sol_p_bar_clus[t,n] for t in T, n in N)
    sol_OF_clus[4] = sum(CO[t]*sol_p_clus[t,ts,n]*CS_P[ts] for t in T, ts in TS, n in N)
    sol_OF_clus[5] = sum(CS*sol_σ_clus[ts,n]*CS_P[ts] for ts in TS,n in N)
    sol_OF_clus[6] = sum(KIT[lo]*sol_f_bar_clus[lo] for lo in 1:length(L))
    sol_OF_clus[7] = sum(KAT*sol_f_bar_clus[lo] for lo in 1:length(L))
    sol_OF_clus[8] = sum(KT*abs(sol_f_clus[ts,lo]) for ts in TS, lo in 1:length(L))
    sol_OF_clus[9] = sum(KIS*sol_s_bar_clus[n] for n in N)
    sol_OF_clus[10] = sum(KS*(sol_charg_clus[ts,n] + sol_discharg_clus[ts,n])*CS_P[ts] for ts in TS, n in N)

    return sol_p_bar_clus, sol_p_clus, sol_σ_clus, sol_OF_clus,
    sol_f_bar_clus , sol_f_clus, sol_s_bar_clus , sol_s_clus,
    sol_charg_clus, sol_discharg_clus
end

################################################################################
### NOTE: Data-driven functions
################################################################################

### Find critical points (p-percentile)
function find_crit_DWS(
    pD::Float64,
    pWS::Float64,
    D::Array{Float64},
    W::Array{Float64},
    S::Array{Float64},
    nodes::Int64
    )
    # pD: Percentile for controlling the worst cases of demand
    # pWS: Percentile for controlling the worst cases of WS
    # D/W/S: Series
    # RP: use of rep. days analysis?
    # nRP: Number of representative days needed
    # d: Days array
    # nodes: # of nodes

    # Error control (probability space)
    if ~(0 <= pD <= 1 && 0 <= pWS <= 1)
        println("Error in probability space!")
        return
    end
    # Creating ranges accordingly to percentiles given
    range_WS = sum(W[:,i] .+ S[:,i] for i in 1:nodes) .<= quantile(sum(W[:,i] .+ S[:,i] for i in 1:nodes),pWS)
    range_D = sum(D[:,i] for i in 1:nodes) .>= quantile(sum(D[:,i] for i in 1:nodes),pD)
    range_DWS = (range_D .* range_WS) .> 0

    # Return with the percentiles used
    range_D_ret = (pD,range_D)
    range_WS_ret = (pWS,range_WS)
    range_DWS_ret = (pD, pWS, range_DWS)

    return range_D_ret, range_WS_ret, range_DWS_ret
end

### Generate
function crit_RP(
    pD::Float64,
    pWS::Float64,
    D::Array{Float64},
    W::Array{Float64},
    S::Array{Float64},
    nodes::Int64,
    nRP::Int64,
    d::Array{Int64},
    pace::Float64,
    max_load::Vector{Float64},
    dem_inc::Vector{Float64}
    )
    #### Function made to select critical points accordingly to a minimal percentile
    #### for demand (pD) and a maximum percentile for W + S (pWS). The algorithm
    #### tries to find a maximum of representative periods nRP as specified in
    #### in the input.

    ### Computing the values using the input pD / pWS
    (range_D_ret, range_WS_ret, range_DWS_ret) = find_crit_DWS(pD,pWS,D.*max_load'.*dem_inc',W,S,nodes)
    crit_RP_ret = unique(d[findall(x -> x > 0,range_DWS_ret[3])])  # Number of days in the original

    ### Methodology for representative days

    ### Initialization
    pD_aux = copy(pD)                           # Percentile to find the critical points in Demand
    pWS_aux = copy(pWS)                         # Percentile to find the critical points in WS
    crit_RP_aux = copy(crit_RP_ret)             # Rep. periods using pD_aux and pWS_aux
    (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)

    # Test to see if a reduction on the nRP is needed
    if length(crit_RP_ret) >= nRP
        stop_aux = 0                            # Variable to stop the while loop
        while stop_aux < 1
            # Crop a quatile window equally reducing pWS and increasing pD
            pD_aux = pD_aux + pace              # Increase D percentile cut
            pWS_aux = pWS_aux - pace            # Decrease WS percentile cut
            # Find the ranges counting with the full demand
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            # Momentaneous RP
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            if length(crit_RP_aux) < nRP
                pD_aux = pD_aux - pace
                pWS_aux = pWS_aux + pace
                stop_aux = 1
            end
            # Updating the ranges after finding the first tight window for pD and pWS
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            # Update RP
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
        end

        # Adjustments in the case the pace was too high for the threshold
        stop_aux = 0
        while stop_aux < 1
            # Ok
            if length(crit_RP_aux) == nRP
                stop_aux = 1

            # Number of RP different than nRP
            else
                println(string("\nAdjustments needed when nRP is ",nRP," in ",now()))
                stop_aux2 = 0                           # Variable to control the next while loop
                n_int = 0                               # Number of iterations to control the time running the while loops
                low_pace = 0                            # Pace lower bound
                up_pace = copy(pace)                    # Pace upper bound
                pace_gap = up_pace - low_pace           # Pace difference to control the number of loops needed
                new_pace = pace/2                       # Updated pace
                while stop_aux2 < 1 && n_int < 1000
                    n_int += 1
                    println(string("\npace gap is: ",pace_gap))
                    pD_aux += new_pace
                    pWS_aux -= new_pace

                    #Reassess the ranges
                    (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
                    #Recalculate RP
                    crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])

                    if length(crit_RP_aux) == nRP
                        # Now we increase the criteria being necessary to match the number of days set by nRP
                        stop_aux2 = 1

                    else
                        println(" / Crit_days = ",length(crit_RP_aux))
                        # Back to the last step
                        pD_aux -= new_pace
                        pWS_aux += new_pace

                        if length(crit_RP_aux) > nRP
                            low_pace = copy(new_pace)
                            new_pace += (up_pace-new_pace)/2
                        else
                            up_pace = copy(new_pace)
                            new_pace -= (new_pace-low_pace)/2
                        end
                    end
                    pace_gap = up_pace - low_pace           # Pace difference to control the number of loops needed
                end
                stop_aux = 1
            end

        end

        if length(crit_RP_aux) != nRP
            println("\nNumber of days still not the same as nRP")

            # # Generate days when the pace is equal to the upper bound
            # pD_aux += up_pace
            # pWS_aux -= up_pace
            # #Reassess the ranges
            # (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            # #Recalculate RP
            # crit_RP_aux_up = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            # pD_aux -= up_pace
            # pWS_aux += up_pace
            #
            # # Generate days when the pace is equal to the lower bound
            # pD_aux += low_pace
            # pWS_aux -= low_pace
            # #Reassess the ranges
            # (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            # #Recalculate RP
            # crit_RP_aux_low = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            # pD_aux -= low_pace
            # pWS_aux += low_pace

            pD_hold = pD_aux + up_pace
            pWS_hold = pWS_aux - up_pace
            (range_D_hold, range_WS_hold, range_DWS_hold) = find_crit_DWS(pD_hold,pWS_hold,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_hold = unique(d[findall(x -> x > 0,range_DWS_hold[3])])
        else
            pD_hold = copy(pD_aux)
            pWS_hold = copy(pWS_aux)
            (range_D_hold, range_WS_hold, range_DWS_hold) = find_crit_DWS(pD_hold,pWS_hold,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_hold = unique(d[findall(x -> x > 0,range_DWS_hold[3])])
        end

        ## Controlling if number of RP is higher than specified
        if length(crit_RP_hold) > nRP
            crit_RP_hold = copy(crit_RP_hold[1:nRP])
        end

        # Trying to decrease W+S percentile
        stop_aux = 0
        while stop_aux < 1
            # Update pD
            pD_aux = pD_aux + pace
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            if length(crit_RP_aux) != length(crit_RP_hold)
                pD_aux = pD_aux - pace
                stop_aux = 1
            end
        end

        # Trying to increase the Demand percentile
        stop_aux = 0
        while stop_aux < 1
            # Update pWS
            pWS_aux = pWS_aux - pace
            (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
            crit_RP_aux = unique(d[findall(x -> x > 0,range_DWS_aux[3])])
            if length(crit_RP_aux) != length(crit_RP_hold)
                pWS_aux = pWS_aux + pace
                stop_aux = 1
            end
        end
    end

    (range_D_aux, range_WS_aux, range_DWS_aux) = find_crit_DWS(pD_aux,pWS_aux,D.*max_load'.*dem_inc',W,S,nodes)
    crit_RP_aux_ret = unique(d[findall(x -> x > 0,range_DWS_aux[3])])

    # Return with the percentiles used
    range_D_aux_ret = range_D_aux
    range_WS_aux_ret = range_WS_aux
    range_DWS_aux_ret = range_DWS_aux

    if crit_RP_hold != crit_RP_aux_ret
        println("\n Diferent RP! Check code. \n")
    end

    return range_D_ret, range_WS_ret, range_DWS_ret, crit_RP_ret,
    range_D_aux_ret, range_WS_aux_ret, range_DWS_aux_ret, crit_RP_aux_ret
end

### Execut RDCCP approach
function run_RDCCP_3D(
    pD::Float64,
    pWS::Float64,
    pace::Float64,
    RP::Int64,
    dist_m::Union{Float64,Int64},
    D::Matrix{Float64},
    W::Matrix{Float64},
    S::Matrix{Float64},
    nodal::Bool,
    lines::Vector{Array{Int64,1}},
    LT::Vector{Int64},
    K::Vector{Float64},
    KA::Matrix{Float64},
    CO::Vector{Float64},
    CS::Float64,
    GCN::Matrix{Float64},
    SC::Float64,
    KIT::Matrix{Float64},
    KT::Float64,
    T_effic::Float64,
    KIS::Float64,
    KS::Float64,
    S_effic_c::Vector{Float64},
    S_effic_d::Vector{Float64},
    S0::Vector{Float64},
    RES::Float64,
    max_load::Vector{Float64},
    dem_inc::Vector{Float64}
    )

    nodes = 3

    ### Input
    RDC_CP_centroids_aux = zeros(8760,3,3)
    if nodal
        (range_D_SP, range_WS_SP, range_DWS_SP, RD_SP,
        range_D_red_SP, range_WS_red_SP, range_DWS_red_SP, RD_red_SP) =
        crit_RP(pD,pWS,D[:,1],W[:,1],S[:,1],1,RP,d,pace,max_load,dem_inc)

        (range_D_DK, range_WS_DK, range_DWS_DK, RD_DK,
        range_D_red_DK, range_WS_red_DK, range_DWS_red_DK, RD_red_DK) =
        crit_RP(pD,pWS,D[:,2],W[:,2],S[:,2],1,RP,d,pace,max_load,dem_inc)

        (range_D_DE, range_WS_DE, range_DWS_DE, RD_DE,
        range_D_red_DE, range_WS_red_DE, range_DWS_red_DE, RD_red_DE) =
        crit_RP(pD,pWS,D[:,3],W[:,3],S[:,3],1,RP,d,pace,max_load,dem_inc)

        RD_red_nodal = unique(sort(vcat(RD_red_SP,RD_red_DK,RD_red_DE)))

        (RDC_CP_clusters,RDC_CP_centroids) = RDC_CP(RD_red_nodal,dist_m,D_d,W_d,S_d)

        ### Forming the DWS series to be analyzed
        RDC_CP_centroids_aux[:] = copy(RDC_CP_centroids[:])
    else
        (range_D, range_WS, range_DWS, RD,
        range_D_red, range_WS_red, range_DWS_red, RD_red) =
        crit_RP(pD,pWS,D,W,S,nodes,RP,d,pace,max_load,dem_inc)

        (RDC_CP_clusters,RDC_CP_centroids) = RDC_CP(RD_red,dist_m,D_d,W_d,S_d)

        ### Forming the DWS series to be analyzed
        RDC_CP_centroids_aux[:] = copy(RDC_CP_centroids[:])
    end

    for i in 1:nodes
        ## Adjusting series to have the same mean (taking care of values higher than 1)
        RDC_CP_centroids_aux[:,i,1] = RDC_CP_centroids_aux[:,i,1].*(mean(D[:,i])/mean(RDC_CP_centroids_aux[:,i,1]))
        RDC_CP_centroids_aux[:,i,2] = RDC_CP_centroids_aux[:,i,2].*(mean(W[:,i])/mean(RDC_CP_centroids_aux[:,i,2]))
        RDC_CP_centroids_aux[:,i,3] = RDC_CP_centroids_aux[:,i,3].*(mean(S[:,i])/mean(RDC_CP_centroids_aux[:,i,3]))
        for j in 1:3        #DWS
            # Number of values higher than 1
            aux = sum(RDC_CP_centroids_aux[:,i,j].>1)
            while aux>0
                # Amount exceding 1
                aux2 = sum(RDC_CP_centroids_aux[RDC_CP_centroids_aux[:,1,3].>1,1,3])-sum(RDC_CP_centroids_aux[:,1,3].>1)
                # Updating high values to 1
                RDC_CP_centroids_aux[RDC_CP_centroids_aux[:,i,j].>1,i,j] .= 1
                # Values lower than 1
                aux3 = sum(RDC_CP_centroids_aux[:,1,3].<1)
                # Distributing the excess
                RDC_CP_centroids_aux[RDC_CP_centroids_aux[:,i,j].<1,i,j] .=
                RDC_CP_centroids_aux[RDC_CP_centroids_aux[:,i,j].<1,i,j] .+ aux2/aux3
                aux = sum(RDC_CP_centroids_aux[:,i,j].>1)
            end
        end
    end

    ### Creating objects to use clus optimization
    ## Sizes of clusters (equals to the number of days assigned to the RD)
    RDC_sizes = zeros(Int,24*RP)
    ## Number of clusters
    RDC_n_clusters_y = 24*collect(1:365)

    ## DWS series with 24*RP representative hours
    RDC_centroids_DWS_clus = Array{Float64}(undef,24*RP,3,3)          # #clus / node / DWS

    ## Forming clusters
    if nodal
        for i in 1:RP
            RDC_centroids_DWS_clus[(i-1)*24 + 1:i*24,:,:] = RDC_CP_centroids_aux[(RD_red_nodal[i]-1)*24 + 1:RD_red_nodal[i]*24,:,:]
        end
    else
        for i in 1:RP
            RDC_centroids_DWS_clus[(i-1)*24 + 1:i*24,:,:] = RDC_CP_centroids_aux[(RD_red[i]-1)*24 + 1:RD_red[i]*24,:,:]
        end
    end

    ## Computing sizes
    for i in 1:RP
        RDC_sizes[(i-1)*24 + 1:i*24] .= Int(sum(RDC_CP_clusters.==i))
    end

    # ### Optimize (as int)
    # (s_RDCCP_Gen_Inv,
    # s_RDCCP_Gen,
    # s_RDCCP_Shed,
    # s_RDCCP_Opt,
    # s_RDCCP_Trans_Inv,
    # s_RDCCP_Trans,
    # s_RDCCP_Stor,
    # s_RDCCP_Stor_Inv,
    # s_RDCCP_Charg,
    # s_RDCCP_Discharg) = CTEM_int(nodes, lines, LT, K, KA, CO, CS, GCN, SC, KIT, KT,
    # T_effic, KIS, KS, S_effic_c, S_effic_d, S0, RES,
    # RDC_CP_centroids_aux[:,:,1],
    # RDC_CP_centroids_aux[:,:,2],
    # RDC_CP_centroids_aux[:,:,3],
    # max_load, dem_inc)

    ### Optimize (as clus)
    @time (s_RDCCP_Gen_Inv,
    s_RDCCP_Gen,
    s_RDCCP_Shed,
    s_RDCCP_Opt,
    s_RDCCP_Trans_Inv,
    s_RDCCP_Trans,
    s_RDCCP_Stor_Inv,
    s_RDCCP_Stor,
    s_RDCCP_Charg,
    s_RDCCP_Discharg) =
    CTEM_clus(nodes, lines, LT, K, KA, CO, CS, GCN, SC, KIT, KT,
    T_effic, KIS, KS, S_effic_c, S_effic_d, S0, RES, RP,
    RDC_centroids_DWS_clus[1:Int.(RDC_n_clusters_y[RP]),:,1],
    RDC_centroids_DWS_clus[1:Int.(RDC_n_clusters_y[RP]),:,2],
    RDC_centroids_DWS_clus[1:Int.(RDC_n_clusters_y[RP]),:,3],
    RDC_n_clusters_y,
    RDC_sizes[1:Int.(RDC_n_clusters_y[RP])],
    max_load,
    dem_inc)

    if nodal
        return RD_red_nodal, RDC_CP_clusters_nodal, RDC_CP_centroids_nodal,
        s_RDCCP_Gen_Inv, s_RDCCP_Gen, s_RDCCP_Shed, s_RDCCP_Opt,
        s_RDCCP_Trans_Inv, s_RDCCP_Trans, s_RDCCP_Stor, s_RDCCP_Stor_Inv,
        s_RDCCP_Charg, s_RDCCP_Discharg
    else
        return RD_red, RDC_CP_clusters, RDC_CP_centroids,
        s_RDCCP_Gen_Inv, s_RDCCP_Gen, s_RDCCP_Shed, s_RDCCP_Opt,
        s_RDCCP_Trans_Inv, s_RDCCP_Trans, s_RDCCP_Stor_Inv, s_RDCCP_Stor,
        s_RDCCP_Charg, s_RDCCP_Discharg
    end
end
