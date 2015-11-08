import re
import collections
import string
NUM_COMMON_WORDS = 1

class parser:
	# self.total_cnt = collections.Counter()
	def __init__(self):
		self.load_common("1-1000.txt",1000)
	def load_common(self,filename,num):
		self.num = num
		self.common = (open(filename,"r")).read().split()[0:num]
		return self.common
	def parse(self, filename):
		conjuctions = ['and', 'but', 'or', 'nor']
		articles =  ['the', 'a', 'an']
		pronouns = ['I', 'me', 'he', 'she', 'his', 'her', 'they', 'their']
		uppercase = string.ascii_uppercase
		output = []
		text_file = open(filename,"r")
		data = text_file.read().replace('\n', ' ')
		char_cnt = collections.Counter()
		total_char = 0
		for c in data:
			total_char += 1
			char_cnt[c] += 1

		period_count = char_cnt['.'] + 1
		excl_count = char_cnt['!'] + 1
		ques_count = char_cnt['?'] + 1
		semi_count = char_cnt[';'] + 1
		dash_count = char_cnt['-'] + 1
		comma_count =  char_cnt[','] + 1
		colon_count  = char_cnt[':'] + 1
		quote1_count = char_cnt["'"] + 1
		quote2_count = char_cnt['"'] + 1
		end_punctuation = period_count + excl_count + ques_count
		other_punctuation = semi_count +  comma_count +  colon_count + dash_count
		quote_count = quote1_count + quote2_count

		wordList = re.sub("[^\w]", " ",  data).split()
		cnt = collections.Counter()
		total_char_np = 0.0
		num_words = 0.0

		for word in wordList:
			total_char_np += len(word)
			num_words += 1
			cnt[word] += 1
		
		total_unique_words = 0.0
		for val in cnt.values():
			if (val == 1):
				total_unique_words += 1
		percent_unique = total_unique_words/num_words
			
		common_words = cnt.most_common(NUM_COMMON_WORDS)

		total_upper = 0.0
		for upper in uppercase:
			total_upper += char_cnt[upper]

		total_conj = 0.0
		for conj in conjuctions:
			total_conj += cnt[conj]

		total_art = 0.0
		for art in articles:
			total_art += cnt[art]

		total_pron = 0.0
		for pron in pronouns:
			total_pron += cnt[pron]

		average_word_length = total_char_np/num_words
		average_sent_length = num_words/(end_punctuation)
		average_conj = total_conj/num_words
		average_art = total_art/num_words
		average_pron =  total_pron/num_words
		average_upper = total_upper/total_char

		output.append(average_sent_length / 50)
		output.append(average_word_length / 10)
		output.append(average_conj * 10)
		output.append(average_art * 10)
		output.append(average_pron * 10)
		output.append(len(cnt)/ num_words)
		output.append(num_words / (50 * other_punctuation))
		output.append(num_words / (1000 * quote_count))
		output.append(num_words / (100 * semi_count))
		output.append(num_words / (1000 * colon_count))
		output.append(num_words / (1000 * dash_count))
		output.append(num_words / (100 * comma_count))
		for x in range(0,NUM_COMMON_WORDS):
			output.append(common_words[x][1] *10 / num_words)
		for i in range(0,self.num):
			output.append(cnt[self.common[i]] * (((i//10) + 1) * 10) / num_words)	
		return output
		# for o in output:
		# 	print(o)

def main():
	prsr = parser()
	print(prsr.parse("docs2/shakespeare/2.txt"))
	print(prsr.parse("docs2/twain/2.txt"))
	print(prsr.parse("docs2/dickens/2.txt"))
	print(prsr.parse("docs2/doyle/2.txt"))
	print(prsr.parse("docs2/milton/2.txt"))





# main()