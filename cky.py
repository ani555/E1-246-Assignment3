import nltk
from nltk.corpus import treebank
from nltk.tree import Tree
import nltk.treetransforms as ttf
from sklearn.model_selection import train_test_split
import re
import argparse
import pickle
import time
import os
import json

parser = argparse.ArgumentParser(description='CKY')
parser.add_argument('--mode', dest='mode', help='parse or test', required=True)
parser.add_argument('--sentence', nargs='+', help='the sentence to be parsed')
parser.add_argument('--data_dir', dest='data_dir', default='data/', required=False)
parser.add_argument('--log_dir', dest='log_dir', default='logs/', required=False)
parser.add_argument('-draw_tree', help='draw the parsed tree', action='store_true')
parser.add_argument('--config_file', dest='config_file', default='config.json', help='config file name with path')

args = parser.parse_args()


if args.mode=='parse':
	if args.sentence==None:
		parser.error('please provide the sentence to be parsed using --sentence flag')



def baseCategory(t):

	# helper function to convert a node to it's base category - taken from the help document posted as a part of the assignment
	# source: http://idiom.ucsd.edu/~rlevy/teaching/2015winter/lign165/lectures/lecture21/lecture21_tree_search_and_pcfgs.pdf 
	if isinstance(t, Tree):
		m = re.match("^(-[^-]+-|[^-=]+)",t.label())
		if m == None:
			print(t.label())
		return(m.group(1))
	else:
		return(t)


def labels_to_base_category(tree):

	children = []
	for t in tree:
		if isinstance(t, Tree):
			children.append(labels_to_base_category(t))
		else:
			children.append(t.lower())
	label = baseCategory(tree) if isinstance(tree, Tree) else tree
	return Tree(label, children)

def add_to_dict(dict_, rule, count):
	
	if rule not in dict_:
		dict_[rule] = count
	else:
		dict_[rule] += count
	

def extract_grammar(root, grammar, pre_terminals):
	
	# populates the grammar dict with rules from the current tree
	root.chomsky_normal_form(horzMarkov=2)
	queue = []
	queue.append(root)
	
	while len(queue)>0:
		
		curr_node = queue.pop(0)
	
		if isinstance(curr_node, nltk.tree.Tree):
			
			# collect rules of form A->B where B can have 1 or 2 elements
			B = []
			for i in range(len(curr_node)):
				
				if isinstance(curr_node[i], nltk.Tree):                   
					queue.append(curr_node[i])
					B.append(curr_node[i].label())
					
				elif isinstance(curr_node[i], str):
					B.append(curr_node[i])
					add_to_dict(pre_terminals, (curr_node.label(), curr_node[i]),1)
			
			rule = (curr_node.label(), *B)
			add_to_dict(grammar, rule, 1)  


def convert_to_base_category(tree_bank):
	for i,tree in enumerate(tree_bank):
		tree_bank[i] = labels_to_base_category(tree)
	return tree_bank

def get_grammar(tree_bank):
	
	grammar = {}
	pre_terminals = {}
	for tree in tree_bank:
		extract_grammar(tree, grammar, pre_terminals)
	return grammar, pre_terminals

def process_tree_bank(tree_bank):

	tree_bank = list(tree_bank)
	for i,tree in enumerate(train_bank):
		train_bank[i] = labels_to_base_category(tree)
	return tree_bank

def calc_probabilities(grammar):
	# tag_probs are probabilities of tags of type count(A)/count(.)
	# rule_probs are probabilities of the type count(A->B,C)/count(A) or count(A->B)/count(A)
	tag_count = {} 
	
	for rule, count in grammar.items():
		add_to_dict(tag_count, rule[0], count)

	rule_probs = {rule:count/tag_count[rule[0]] for rule, count in grammar.items()}
	N = sum(tag_count.values())
	tag_probs = {tag:count/N for tag, count in tag_count.items()}
	
	return tag_probs, rule_probs, tag_count

def get_smoothed_prob(rule, tag_probs, rule_probs, smoothing='backoff', alpha=0.8):

	# rule_probs: count(A->B,C)/count(A) or count(A->B)/count(A)
	# tag_probs: count(A)/count()
	
	if smoothing == 'interpolation':
		term1 = alpha * rule_probs[rule] if rule in rule_probs else 0
		term2 = (1-alpha) * tag_probs[rule[0]] if rule[0] in tag_probs else 0
		return term1+term2
	elif smoothing == 'backoff':
		if rule in rule_probs:
			return rule_probs[rule]
		else:
			return alpha*tag_probs[rule[0]]	
	else:
		raise Exception('Not a valid smoothing method')

def cky(words, grammar, tag_probs, rule_probs, smoothing='backoff', alpha=0.8):
	num_words = len(words)
	non_terms = list(tag_probs.keys())
	num_nonterms = len(tag_probs)
	score = [[[0 for i in range(num_nonterms)] for j in range(num_words+1)] for k in range(num_words+1)]
	bptr = [[[0 for i in range(num_nonterms)] for j in range(num_words+1)] for k in range(num_words+1)]
	tag2idx = {tag:i for i, tag in enumerate(non_terms)}
	
	non_terms_filled, non_terms_left = ([], [])
	# fill up the (non-term->term) rules in span (i, i+1) for level 0
	for i in range(num_words):
		for A in non_terms:
			rule = (A, words[i])
			if rule in grammar:
				score[i][i+1][tag2idx[A]] = get_smoothed_prob(rule, tag_probs, rule_probs, smoothing, alpha)
				non_terms_filled.append(A)
			else:
				non_terms_left.append(A)
#         print(non_terms_filled) 
		non_terms_filled_tmp = non_terms_filled[:]
		non_terms_left_tmp = non_terms_left[:]
		added = True
		while added:
			added = False
			for A in non_terms_left:
				for B in non_terms_filled:
					rule = (A, B)
					if score[i][i+1][tag2idx[B]]>0 and rule in grammar:
						prob = get_smoothed_prob(rule, tag_probs, rule_probs, smoothing, alpha)*score[i][i+1][tag2idx[B]]
						if prob > score[i][i+1][tag2idx[A]]:
							score[i][i+1][tag2idx[A]] = prob
							bptr[i][i+1][tag2idx[A]] = B
							added = True
							if A in non_terms_left_tmp:
								non_terms_left_tmp.remove(A)
							if B in non_terms_filled_tmp:
								non_terms_filled_tmp.remove(B)
			# print(non_terms_filled)
			# print(non_terms_left)
			non_terms_filled = non_terms_filled_tmp[:]
			non_terms_left = non_terms_left_tmp[:]

	a_to_bc_rules = [rule for rule in rule_probs.keys() if len(rule)==3]
	# print(len(a_to_bc_rules))
	for span in range(2, num_words+1):
		for begin in range(num_words+1-span):
			end = begin + span
			non_terms_filled = []
			# for a given span vary the splits and find the best split for a rule for level>0
			for split in range(begin+1, end):
				for rule in a_to_bc_rules:
#                     print(rule)
					A, B, C = rule
					prob = score[begin][split][tag2idx[B]]*score[split][end][tag2idx[C]]*get_smoothed_prob(rule, tag_probs, rule_probs, smoothing, alpha)
					# if prob>0:
					#     print(prob)
					if prob > score[begin][end][tag2idx[A]]:
						# print('updated {} to {} rule {}'.format(begin, end, A))
						score[begin][end][tag2idx[A]] = prob
						bptr[begin][end][tag2idx[A]] = (split, B, C)
						non_terms_filled.append(A)
			
			# check for single edge possibilities            
			added = True
			while added:
				added=False
				for A in non_terms:
					for B in non_terms_filled:
						rule = (A,B)
						if rule in grammar:
							prob =  get_smoothed_prob(rule, tag_probs, rule_probs, smoothing, alpha)*score[begin][end][tag2idx[B]]
							if prob > score[begin][end][tag2idx[A]]:
								score[begin][end][tag2idx[A]] = prob
								bptr[begin][end][tag2idx[A]] = B
								added = True
							
	return score, bptr

def print_scores(score, tag_probs, num_words, idx2tag):

	
	for i in range(num_words):
		for j in range(num_words):
			for k in range(len(tag_probs)):
				if score[i][j][k]>0:
					print("span i: {}, j: {}, nonterm: {}, score: {:.29f}".format(i,j,idx2tag[k], score[i][j][k]))                  



def build_tree_util(curr_id, begin, end, bptr, tag2idx, idx2tag, leaves):

	# print(begin,end, idx2tag[curr_id])
	t = bptr[begin][end][curr_id]
	# print(idx2tag[curr_id], t)
	# print(begin, end,idx2tag[curr_id])
	if t==0 and begin == end-1:	
		return Tree(idx2tag[curr_id],[leaves[begin]])
	
	children = []
	if isinstance(t, tuple) and len(t) == 3:
		split, B, C = t
		# print("triple:",split, B, C)
		children.append(build_tree_util(tag2idx[B], begin, split, bptr, tag2idx, idx2tag, leaves))
		children.append(build_tree_util(tag2idx[C], split, end, bptr, tag2idx, idx2tag, leaves))
	else:
		if t!=0 and tag2idx[t]!=0:
			# print("single edge:",t)
			children.append(build_tree_util(tag2idx[t], begin, end, bptr, tag2idx, idx2tag, leaves))
	
	return Tree(idx2tag[curr_id], children)


def build_tree(score, bptr, tag2idx, idx2tag, leaves):

	start_id = tag2idx['S']
	begin = 0
	end = len(bptr)-1
	tree = build_tree_util(start_id, begin, end, bptr, tag2idx, idx2tag, leaves)
	ttf.un_chomsky_normal_form(tree)
	return tree

def extend_grammar(words, grammar, pre_terminals):

	pre_keys = list(pre_terminals.keys())
	pre_words = [rule[1] for rule in pre_keys]
	# print(pre_words)
	words_not_in_grammar = [word for word in words if word not in pre_words]
	# print(words_not_in_grammar)
	for rule in pre_keys:
		for word in words_not_in_grammar:
			pre_rule = (rule[0], word)
			add_to_dict(grammar, pre_rule, 1)
			add_to_dict(pre_terminals, pre_rule, 1)

def get_processed_data():
	
	bank = treebank.parsed_sents()
	train_bank, test_bank = train_test_split(bank, test_size=0.2)
	train_bank = list(train_bank)
	test_bank = list(test_bank)
	train_bank = convert_to_base_category(train_bank)
	test_bank = convert_to_base_category(test_bank)

	return train_bank, test_bank
	

def save_obj(obj, file_name):

	with open(file_name, 'wb') as f:
		pickle.dump(obj,f)

def load_obj(file_name):

	with open(file_name, 'rb') as f:
		obj = pickle.load(f)
	return obj

def write_to_file(trees, file_name):

	with open(file_name, 'w') as f:
		for tree in trees:
			f.write(tree)
			f.write('\n')

def main():

	if args.mode=='train':	
		
		if not os.path.exists(args.data_dir):
			os.makedirs(args.data_dir)
		
		train_bank, test_bank = get_processed_data()
		grammar, pre_terminals = get_grammar(train_bank)
		tag_probs, rule_probs, tag_counts = calc_probabilities(grammar)		
		idx2tag = {i:tag for i, tag in enumerate(tag_probs.keys())}
		tag2idx = {tag:i for i, tag in enumerate(tag_probs.keys())}
		save_obj(train_bank, os.path.join(args.data_dir,'train.pkl'))
		save_obj(test_bank, os.path.join(args.data_dir,'test.pkl'))
		save_obj((grammar, pre_terminals), os.path.join(args.data_dir,'grammar.pkl'))
		save_obj((tag_probs, rule_probs, tag_counts), os.path.join(args.data_dir,'probabilities.pkl'))
		save_obj((idx2tag, tag2idx), os.path.join(args.data_dir,'mappings.pkl'))


	else:

		train_bank = load_obj(os.path.join(args.data_dir,'train.pkl'))
		test_bank = load_obj(os.path.join(args.data_dir,'test.pkl'))
		grammar, pre_terminals = load_obj(os.path.join(args.data_dir,'grammar.pkl'))
		tag_probs, rule_probs, tag_counts = load_obj(os.path.join(args.data_dir,'probabilities.pkl'))
		idx2tag, tag2idx = load_obj(os.path.join(args.data_dir, 'mappings.pkl'))
		# print(rule_probs[('NP','NP', 'SBAR')], rule_probs[('VP','VBD','SBAR')])
		# print(tag_probs['-NONE-'], tag_probs['NN'])
		if not os.path.exists(args.log_dir):
			os.makedirs(args.log_dir)
		
		with open(args.config_file, 'r') as f:
			config = json.load(f)
		
		if args.mode == 'parse':

			words = nltk.word_tokenize(' '.join(args.sentence).lower())
			extend_grammar(words, grammar, pre_terminals)
			num_pre_terminals = len(pre_terminals)
			score, bptr = cky(words, grammar, tag_probs, rule_probs, smoothing=config['smoothing'], alpha=config['alpha'])
			parsed_tree = build_tree(score, bptr, tag2idx, idx2tag, words)
			write_to_file([parsed_tree.pformat()], os.path.join(args.log_dir,'parsed_tree.tst'))
			print(parsed_tree.pformat())
			if args.draw_tree:
				parsed_tree.draw()

		elif args.mode == 'test':
		
	
			num_eval = config['num_eval']
			parsed_trees = []
			gold_trees = []

			start = time.time()
			for i in range(min(num_eval, len(test_bank))):

				words = test_bank[i].leaves()
				num_words = len(words)
				extend_grammar(words, grammar, pre_terminals)
				num_pre_terminals = len(pre_terminals)
				score, bptr = cky(words, grammar, tag_probs, rule_probs, smoothing=config['smoothing'], alpha=config['alpha'])
				parsed_tree = build_tree(score, bptr, tag2idx, idx2tag, words)
				print('parsed_tree: {}, gold_tree: {}'.format(len(parsed_tree.leaves()), num_words))
				print('finished {}'.format(i))
				flat_parse_tree = ' '.join(parsed_tree.pformat().split())
				flat_gold_tree = ' '.join(test_bank[i].pformat().split())
				parsed_trees.append(flat_parse_tree)
				gold_trees.append(flat_gold_tree)
		
			write_to_file(gold_trees, os.path.join(args.log_dir,'gold_trees.gld'))
			write_to_file(parsed_trees, os.path.join(args.log_dir,'parsed_trees.tst'))
			end = time.time()
			print('finished parsing {} sentences in {} secs.'.format(num_eval, end-start))




if __name__ == '__main__':
	main()