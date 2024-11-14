function angle=convto360(omega)
    while omega>360 || omega<0
        if omega<0
            omega = 360+omega;
        elseif omega>360
            omega = omega-360;
        end
    end
    angle = omega;
end

