
start=100000
end=1000000
step=100000

for N in $(seq $start $step $end); do
    TimesEj1Cpu=$(./cuda/Ejercicio1/cpu $N)
    TimesEj1Gpu=$(./cuda/Ejercicio1/gpu $N)
    echo "$TimesEj1Cpu $TimesEj1Gpu"
done
