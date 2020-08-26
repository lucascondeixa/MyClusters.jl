include("myclusters_github.jl")
using Test

## Load instances
function load_cases()
    cd("..")
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
    cd("test")
    # Days accounting
    for i in 1:8760
        d[i] = Int(ceil(i/24))
    end
end

## Test of kmeans over DWS with k=10
function test_kmeans()
    cd("..")
    cd("examples")
    a = kmeans3D(true,
        10,
        3,
        D[:,1],
        W[:,1],
        S[:,1]
    )
    aux = readdlm("kmeans3D_10_first_s.txt")
    if aux[:] != Int.(sort(a[3][10,1:10]))
        println("\nDifferent seed, check code!")
        return
    end
    # writedlm("wogrin3D_new_10.txt",a[1][1:10,:])
    aux = readdlm("kmeans3D_10.txt")
    cd("..")
    cd("test")
    @test a[1][1:10,:] == aux
end

## Test of HCnD using a reduced DWS[1:50,:]
function test_HCnD_WD()
    cd("..")
    cd("examples")
    a = HCnD(1,1,D[1:50,:],W[1:50,:],S[1:50,:])
    # writedlm("HCnD_50.txt",a[1])
    aux = readdlm("HCnD_WD_50.txt")
    cd("..")
    cd("test")
    @test a[1] == aux
end

## Test of HCnD using a reduced DWS[1:50,:]
function test_HCnD_ED()
    cd("..")
    cd("examples")
    a = HCnD(1,2,D[1:50,:],W[1:50,:],S[1:50,:])
    # writedlm("HCnD_ED_50.txt",a[1])
    aux = readdlm("HCnD_ED_50.txt")
    cd("..")
    cd("test")
    @test a[1] == aux
end

## Test of HCnD using a reduced DWS[1:50,:]
function test_crit_RP()
    cd("..")
    cd("examples")
    a = crit_RP(pD,pWS,D,W,S,nodes,RP,d,pace,max_load,dem_inc)
    # writedlm("crit_RP_50.txt",a[8])
    aux = readdlm("crit_RP_50.txt")
    cd("..")
    cd("test")
    @test a[8] == aux[:]
end

## Test of HCnD using a reduced DWS[1:50,:]
function test_find_crit_DWS()
    cd("..")
    cd("examples")
    a = find_crit_DWS(pD_crit,pWS_crit,D,W,S,nodes)
    # writedlm(string("find_crit_DWS_pD_",pD_crit,"_pWS_",pWS_crit,".txt"),a[3][3])
    aux = readdlm(string("find_crit_DWS_pD_",pD_crit,"_pWS_",pWS_crit,".txt"))
    cd("..")
    cd("test")
    @test a[3][3] == aux[:]
end
