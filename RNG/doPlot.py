import roadrunner

rr = roadrunner.RoadRunner(r"D:\RNG\RNG\models\LargeHanekomNetworks\sbml\0.sbml")
rr.simulate(0, 10000, 10000)
rr.plot()
