import roadrunner

rr = roadrunner.RoadRunner(r"D:\RNG\RNG\models\LargeHanekomNetworks\sbml\7.sbml")
rr.simulate(0, 10000, 10000)
rr.plot()