import info_parser as  ip

def main():
	prsr = ip.parser()
	c1 = prsr.load_common("1-1000.txt",100)
	c2 = prsr.load_common("1-1000.txt",100)
	for i in range(0,100):
		print(i,c1[i])
		print(i,c2[i])



main()