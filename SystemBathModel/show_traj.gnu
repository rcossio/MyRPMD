#set xrange [-3e-10:3e-10]
#set yrange [-3e-10:3e-10]
#set zrange [-3e-10:3e-10]

sp "tmp.dat" u 2:3:4:1    w l palette  t "Bead 1", \
   ""        u 5:6:7:1    w l palette  t "Bead 2", \
   ""        u 8:9:10:1   w l palette  t "Bead 3", \
   ""        u 11:12:13:1 w l palette  t "Bead 4"

pause -1
