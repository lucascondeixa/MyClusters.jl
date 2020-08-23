cd(@__DIR__)
push!(LOAD_PATH, dirname(@__DIR__))
include("myclusters_github.jl")

using Gurobi
using Dates
using DelimitedFiles
# using CPLEX
# using CSV
# using Plots
# using DataFrames
# using Suppressor
# using XLSX


## Test objects
nodes = 3
lines = [[1,3], [2,3]]
Cluss = 1:650
D = Array{Float64}(undef, 8760, nodes)
W = Array{Float64}(undef, 8760, nodes)
S = Array{Float64}(undef, 8760, nodes)
max_load = Array{Float64}(undef, nodes)
dem_inc = Array{Float64}(undef, nodes)
m = zeros(8760)
d = zeros(Int,8760)
pD = 0.01                           # Percentile to find the critical points in Demand
pWS = 0.99                          # Percentile to find the critical points in WS
pace = 0.0005
RP = 50

## Load instances
cases = ["SP", "DK", "DE"]
function load_cases()
    cd("instances")
    for i in 1:nodes
        case = cases[i]
        df = CSV.read(string("Case_","$case","_New.csv"))
        D[:,i] = Float64.(df.Load_mod)
        W[:,i] = Float64.(df.Avail_Win)
        S[:,i] = Float64.(df.Avail_Sol)
        max_load[i] = Float64.(df.Max_Load[1])
        dem_inc[i] = Float64.(df.Dem_Inc[1])
        if i == 1
            global m = Int64.(df.Month)
        end
    end
    cd("..")
    # Days accounting
    for i in 1:8760
        d[i] = Int(ceil(i/24))
    end
end

## Test of kmeans over DWS with k=10
function test_kmeans()
    cd("examples")
    a = kmeans3D(true,
        10,
        3,
        D[:,1],
        W[:,1],
        S[:,1]
    )
    # writedlm("wogrin3D_new_10.txt",a[1][1:10,:])
    aux = readdlm("kmeans3D_10.txt")
    cd("..")
    @assert a[1][1:10,:] == aux
end

## Test of HCnD using a reduced DWS[1:50,:]
function test_HCnD()
    cd("examples")
    a = HCnD(1,1,D[1:50,:],W[1:50,:],S[1:50,:])
    # writedlm("HCnD_50.txt",a[1])
    aux = readdlm("HCnD_50.txt")
    cd("..")
    @assert a[1] == aux
end

## Test of HCnD using a reduced DWS[1:50,:]
function test_crit_RP()
    cd("examples")
    a = crit_RP(pD,pWS,D,W,S,nodes,RP,d,pace,max_load,dem_inc)
    # writedlm("crit_RP_50.txt",a[8])
    aux = readdlm("crit_RP_50.txt")
    cd("..")
    @assert a[8] == aux[:]
end

load_cases()
# test_kmeans()
# test_HCnD()
test_crit_RP()

# End of Code
