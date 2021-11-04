set terminal png size 800,800
set output "Solution.png"

set xlabel "x"

set autoscale

set ylabel "f"

set title "1D Advection Diffusion Equation Solution"

set grid

set style data lines

set yrange [-1.5:1.5]

plot 'solution.dat' using 1:2 lw 2 lc 8 title 'nodes=200,t=0.5'
