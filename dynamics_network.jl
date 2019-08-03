
function bin(θ, I = 6)
    a = [θ - (i-1)*pi/2 for i in 1:I]
    relu.(a - 100*relu.(a .- pi/2))
end
function UB(θ)
    D = Diagonal([1, -1, -2/pi, 2/pi, 1, -1])
    out = D*bin(θ)
    out + [0, pi/2, 0, -1, 0, pi/2]
end