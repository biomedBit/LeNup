-- Troyanskaya's normalization
function troy_norm(p, u)
    local xn = math.log(0.05/0.95)
    local pn = torch.log(torch.cdiv(p,(-p+1)))
    local un
    if type(u) == "number" then
        un = torch.FloatTensor((#p)[1],1):fill(torch.log(u/(-u+1)))
    else
        un = torch.log(torch.cdiv(u,-u+1)):repeatTensor((#p)[1], 1)
    end
    un = un:float() -- probably unnecessary for all future models
    local zn = pn + xn - un
    return (torch.exp(-zn)+1):pow(-1)
end
