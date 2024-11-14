%% Conversion needed for the pretrained NN-beamformers
function newAOA=nnConv(AOA)
    newAOA = azel2phitheta(AOA);
    newAOA(1) = convto360(newAOA(1)+90);
end