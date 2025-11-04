set terminal qt size 1200,900
set multiplot layout 3,1 title "Haar DWT (A and D at multiple scales)" font ",10"

set grid; set key left top
set title "Original"
plot "wave_x.tsv" using 1:2 with lines title "x(t)"

unset key
set title "Approximations (A1..A4)"
plot for [i=1:4] sprintf("wave_A%d.tsv",i) using 1:2 with lines lw (3-i*0.4)

set title "Details (D1..D4)"
plot for [i=1:4] sprintf("wave_D%d.tsv",i) using 1:2 with lines lw (3-i*0.4)

unset multiplot
