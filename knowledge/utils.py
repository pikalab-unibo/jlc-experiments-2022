import re
from psyki.logic import Formula
from psyki.logic.datalog import get_formula_from_string
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog

from knowledge import PATH as KNOWLEDGE_PATH

# Splice junction variables
SPLICE_JUNCTION_FEATURES = ['a', 'c', 'g', 't']
SPLICE_JUNCTION_CLASS_MAPPING = {'ei': 0, 'ie': 1, 'n': 2}
SPLICE_JUNCTION_AGGREGATE_FEATURE = {'a': ('a',),
                                     'c': ('c',),
                                     'g': ('g',),
                                     't': ('t',),
                                     'd': ('a', 'g', 't'),
                                     'm': ('a', 'c'),
                                     'n': ('a', 'c', 'g', 't'),
                                     'r': ('a', 'g'),
                                     's': ('c', 'g'),
                                     'y': ('c', 't')}
# Ad-hoc symbols
VARIABLE_BASE_NAME = 'X'
AND_SYMBOL = ' , '
OR_SYMBOL = ' ; '
NOT_SYMBOL = '¬'
LESS_EQUAL_SYMBOL = ' =< '
PLUS_SYMBOL = ' + '
STATIC_IMPLICATION_SYMBOL = ' <- '
MUTABLE_IMPLICATION_SYMBOL = ' <-- '
STATIC_RULE_SYMBOL = '::-'
MUTABLE_RULE_SYMBOL = ':-'
INDEX_IDENTIFIER = '@'
NOT_IDENTIFIER = 'not'
RULE_DEFINITION_SYMBOLS = (STATIC_RULE_SYMBOL, MUTABLE_RULE_SYMBOL)
RULE_DEFINITION_SYMBOLS_REGEX = '(' + '|'.join(RULE_DEFINITION_SYMBOLS) + ')'


# Ad-hoc parse function for the prior knowledge of the splice junction domain
def parse_splice_junction_clause(rest: str, rhs: str = '', aggregation: str = AND_SYMBOL) -> str:
    def next_index(i: str, indices: list[int], offset: int) -> int:
        new_index: int = int(i) + offset
        modified: bool = False
        while new_index not in indices:
            new_index += 1
            modified = True
        return new_index + previous_holes(indices, indices.index(new_index)) if not modified else new_index

    def previous_holes(l: list[int], i: int) -> int:
        j = 0
        for k in list(range(0, i)):
            if l[k] + 1 != l[k + 1]:
                j += 1
        return j

    def explicit_variables(e: str) -> str:
        result = ''
        for key in SPLICE_JUNCTION_AGGREGATE_FEATURE.keys():
            if key.lower() in e:
                values = [v for v in SPLICE_JUNCTION_AGGREGATE_FEATURE[key]]
                if len(values) > 1:
                    result += AND_SYMBOL.join(
                        NOT_SYMBOL + '(' + re.sub(key.lower(), value.lower(), e) + ')' for value in values)
        return NOT_SYMBOL + '(' + result + ')' if result != '' else e

    for j, clause in enumerate(rest.split(',')):
        index = re.match(INDEX_IDENTIFIER + '[-]?[0-9]*', clause)
        negation = re.match(NOT_IDENTIFIER, clause)
        n = re.match('[0-9]*of', clause)
        if index is not None:
            index = clause[index.regs[0][0]:index.regs[0][1]]
            clause = clause[len(index):]
            clause = re.sub('\'', '', clause)
            index = index[1:]
            rhs += aggregation.join(explicit_variables(
                VARIABLE_BASE_NAME + ('_' if next_index(index, list(range(-30, 0)) + list(range(1, 31)), i) < 0 else '') +
                str(abs(next_index(index, list(range(-30, 0)) + list(range(1, 31)), i))) +
                ' = ' + value.lower()) for i, value in enumerate(clause))
        elif negation is not None:
            new_clause = re.sub(NOT_IDENTIFIER, NOT_SYMBOL, clause)
            new_clause = re.sub('-', '_', new_clause.lower())
            new_clause = re.sub('\)', '())', new_clause)
            rhs += new_clause
        elif n is not None:
            new_clause = clause[n.regs[0][1]:]
            new_clause = re.sub('\(|\)', '', new_clause)
            inner_clause = parse_splice_junction_clause(new_clause, rhs, PLUS_SYMBOL)
            inner_clause = '(' + ('), (').join(e for e in inner_clause.split(PLUS_SYMBOL)) + ')'
            n = clause[n.regs[0][0]:n.regs[0][1] - 2]
            rhs += 'm_of_n(' + n + ', ' + inner_clause + ')'
        else:
            rhs += re.sub('-', '_', clause.lower()) + '()'
        if j < len(rest.split(',')) - 1:
            rhs += AND_SYMBOL
    return rhs


def load_splice_junction_knowledge() -> list[Formula]:
    rules = []
    file = KNOWLEDGE_PATH / 'splice-junction-kb.txt'
    with open(file) as file:
        for raw in file:
            raw = re.sub('\n', '', raw)
            if len(raw) > 0:
                rules.append(raw)
    new_rules = []
    for rule in rules:
        rule = re.sub(r' |\.', '', rule)
        name, op, rest = re.split(RULE_DEFINITION_SYMBOLS_REGEX, rule)
        name = re.sub('-', '_', name.lower())
        rhs = parse_splice_junction_clause(rest)
        if name in SPLICE_JUNCTION_CLASS_MAPPING.keys():
            new_rules.append('class(' + name + ')' + (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL) + rhs)
        new_rules.append(name + '(' + ')' + (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL) + rhs)
    results = []
    term_regex = '[a-z]+'
    variable_regex = VARIABLE_BASE_NAME + '[_]?[0-9]+'
    regex = variable_regex + '[ ]?=[ ]?' + term_regex
    for rule in new_rules:
        tmp_rule = rule
        partial_result = ''
        while re.search(regex, tmp_rule) is not None:
            match = re.search(regex, tmp_rule)
            start, end = match.regs[0]
            matched_string = tmp_rule[start:end]
            ante = tmp_rule[:start]
            medio = matched_string[:re.search(variable_regex, matched_string).regs[0][1]] + \
                    matched_string[re.search(term_regex, matched_string).regs[0][0]:]
            partial_result += ante + medio
            tmp_rule = tmp_rule[end:]
        partial_result += tmp_rule
        results.append(partial_result)
    return [get_formula_from_string(rule) for rule in results]


# Breast cancer utility function
def load_breast_cancer_knowledge() -> list[Formula]:
    knowledge = file_to_prolog(KNOWLEDGE_PATH / 'breast-cancer-kb.pl')
    return prolog_to_datalog(knowledge, trainable=True)
