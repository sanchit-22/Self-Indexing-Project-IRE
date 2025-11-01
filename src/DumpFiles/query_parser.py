"""
Boolean Query Parser and Executor

Implements a parser for Boolean queries with support for:
- Operators: AND, OR, NOT, PHRASE
- Parentheses for grouping
- Correct precedence: PHRASE > NOT > AND > OR
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import defaultdict


class QueryToken:
    """Represents a token in a query."""
    
    def __init__(self, token_type: str, value: str):
        self.type = token_type  # TERM, AND, OR, NOT, PHRASE, LPAREN, RPAREN
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"


class QueryParser:
    """Parser for Boolean queries."""
    
    def __init__(self):
        self.tokens: List[QueryToken] = []
        self.pos = 0
    
    def tokenize(self, query: str) -> List[QueryToken]:
        """
        Tokenize a query string.
        
        Args:
            query: Query string
            
        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        
        while i < len(query):
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue
            
            # Parentheses
            if query[i] == '(':
                tokens.append(QueryToken('LPAREN', '('))
                i += 1
                continue
            
            if query[i] == ')':
                tokens.append(QueryToken('RPAREN', ')'))
                i += 1
                continue
            
            # Quoted terms (PHRASE or TERM)
            if query[i] == '"':
                j = i + 1
                while j < len(query) and query[j] != '"':
                    j += 1
                if j < len(query):
                    term = query[i+1:j]
                    # Check if it's a phrase (multiple words)
                    if ' ' in term:
                        tokens.append(QueryToken('PHRASE', term))
                    else:
                        tokens.append(QueryToken('TERM', term))
                    i = j + 1
                else:
                    # Unclosed quote - treat as term
                    tokens.append(QueryToken('TERM', query[i+1:]))
                    break
                continue
            
            # Check for operators (case-insensitive)
            remaining = query[i:].upper()
            
            if remaining.startswith('AND') and (i + 3 >= len(query) or not query[i+3].isalnum()):
                tokens.append(QueryToken('AND', 'AND'))
                i += 3
                continue
            
            if remaining.startswith('OR') and (i + 2 >= len(query) or not query[i+2].isalnum()):
                tokens.append(QueryToken('OR', 'OR'))
                i += 2
                continue
            
            if remaining.startswith('NOT') and (i + 3 >= len(query) or not query[i+3].isalnum()):
                tokens.append(QueryToken('NOT', 'NOT'))
                i += 3
                continue
            
            # Unquoted term (for backward compatibility)
            j = i
            while j < len(query) and not query[j].isspace() and query[j] not in '()':
                j += 1
            if j > i:
                tokens.append(QueryToken('TERM', query[i:j]))
                i = j
                continue
            
            i += 1
        
        return tokens
    
    def parse(self, query: str) -> 'QueryNode':
        """
        Parse a query string into an AST.
        
        Args:
            query: Query string
            
        Returns:
            Root node of query AST
        """
        self.tokens = self.tokenize(query)
        self.pos = 0
        
        if not self.tokens:
            return TermNode("")
        
        return self.parse_or()
    
    def current_token(self) -> Optional[QueryToken]:
        """Get current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def consume_token(self) -> Optional[QueryToken]:
        """Consume and return current token."""
        token = self.current_token()
        if token:
            self.pos += 1
        return token
    
    def parse_or(self) -> 'QueryNode':
        """Parse OR expression (lowest precedence)."""
        left = self.parse_and()
        
        while self.current_token() and self.current_token().type == 'OR':
            self.consume_token()  # consume OR
            right = self.parse_and()
            left = OrNode(left, right)
        
        return left
    
    def parse_and(self) -> 'QueryNode':
        """Parse AND expression."""
        left = self.parse_not()
        
        while self.current_token() and self.current_token().type == 'AND':
            self.consume_token()  # consume AND
            right = self.parse_not()
            left = AndNode(left, right)
        
        return left
    
    def parse_not(self) -> 'QueryNode':
        """Parse NOT expression."""
        if self.current_token() and self.current_token().type == 'NOT':
            self.consume_token()  # consume NOT
            operand = self.parse_not()  # NOT is right-associative
            return NotNode(operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> 'QueryNode':
        """Parse primary expression (term, phrase, or parenthesized expression)."""
        token = self.current_token()
        
        if not token:
            return TermNode("")
        
        # Parenthesized expression
        if token.type == 'LPAREN':
            self.consume_token()  # consume (
            node = self.parse_or()
            if self.current_token() and self.current_token().type == 'RPAREN':
                self.consume_token()  # consume )
            return node
        
        # Phrase
        if token.type == 'PHRASE':
            self.consume_token()
            return PhraseNode(token.value)
        
        # Term
        if token.type == 'TERM':
            self.consume_token()
            return TermNode(token.value)
        
        # Unexpected token - skip it
        self.consume_token()
        return TermNode("")


class QueryNode:
    """Base class for query AST nodes."""
    
    def evaluate(self, index: Any) -> Set[str]:
        """
        Evaluate the query node against an index.
        
        Args:
            index: The search index
            
        Returns:
            Set of matching document IDs
        """
        raise NotImplementedError


class TermNode(QueryNode):
    """Node representing a single term."""
    
    def __init__(self, term: str):
        self.term = term.lower()
    
    def evaluate(self, index: Any) -> Set[str]:
        """Find documents containing the term."""
        if not self.term or self.term not in index.inverted_index:
            return set()
        return set(index.inverted_index[self.term].keys())
    
    def __repr__(self):
        return f"Term('{self.term}')"


class PhraseNode(QueryNode):
    """Node representing a phrase query."""
    
    def __init__(self, phrase: str):
        self.phrase = phrase
        self.terms = phrase.lower().split()
    
    def evaluate(self, index: Any) -> Set[str]:
        """Find documents containing the exact phrase."""
        if not self.terms:
            return set()
        
        # Get documents containing all terms
        candidate_docs = None
        for term in self.terms:
            if term not in index.inverted_index:
                return set()
            term_docs = set(index.inverted_index[term].keys())
            if candidate_docs is None:
                candidate_docs = term_docs
            else:
                candidate_docs &= term_docs
        
        if not candidate_docs:
            return set()
        
        # Check for exact phrase match using positions
        matching_docs = set()
        for doc_id in candidate_docs:
            if self._has_phrase(index, doc_id):
                matching_docs.add(doc_id)
        
        return matching_docs
    
    def _has_phrase(self, index: Any, doc_id: str) -> bool:
        """Check if document contains the exact phrase."""
        # Get positions for first term
        first_term = self.terms[0]
        if first_term not in index.inverted_index:
            return False
        
        posting = index.inverted_index[first_term].get(doc_id)
        if not posting:
            return False
        
        # Handle different index structures
        if isinstance(posting, list):
            first_positions = posting
        elif isinstance(posting, dict) and 'positions' in posting:
            first_positions = posting['positions']
        else:
            return False
        
        # For each position of first term, check if subsequent terms follow
        for start_pos in first_positions:
            has_phrase = True
            for i, term in enumerate(self.terms[1:], 1):
                if term not in index.inverted_index:
                    has_phrase = False
                    break
                
                term_posting = index.inverted_index[term].get(doc_id)
                if not term_posting:
                    has_phrase = False
                    break
                
                # Get positions for this term
                if isinstance(term_posting, list):
                    term_positions = term_posting
                elif isinstance(term_posting, dict) and 'positions' in term_posting:
                    term_positions = term_posting['positions']
                else:
                    has_phrase = False
                    break
                
                # Check if term appears at expected position
                expected_pos = start_pos + i
                if expected_pos not in term_positions:
                    has_phrase = False
                    break
            
            if has_phrase:
                return True
        
        return False
    
    def __repr__(self):
        return f"Phrase('{self.phrase}')"


class AndNode(QueryNode):
    """Node representing AND operation."""
    
    def __init__(self, left: QueryNode, right: QueryNode):
        self.left = left
        self.right = right
    
    def evaluate(self, index: Any) -> Set[str]:
        """Find documents matching both operands."""
        left_docs = self.left.evaluate(index)
        right_docs = self.right.evaluate(index)
        return left_docs & right_docs
    
    def __repr__(self):
        return f"And({self.left}, {self.right})"


class OrNode(QueryNode):
    """Node representing OR operation."""
    
    def __init__(self, left: QueryNode, right: QueryNode):
        self.left = left
        self.right = right
    
    def evaluate(self, index: Any) -> Set[str]:
        """Find documents matching either operand."""
        left_docs = self.left.evaluate(index)
        right_docs = self.right.evaluate(index)
        return left_docs | right_docs
    
    def __repr__(self):
        return f"Or({self.left}, {self.right})"


class NotNode(QueryNode):
    """Node representing NOT operation."""
    
    def __init__(self, operand: QueryNode):
        self.operand = operand
    
    def evaluate(self, index: Any) -> Set[str]:
        """Find documents not matching the operand."""
        all_docs = set(index.doc_metadata.keys())
        operand_docs = self.operand.evaluate(index)
        return all_docs - operand_docs
    
    def __repr__(self):
        return f"Not({self.operand})"


class BooleanQueryExecutor:
    """Executor for Boolean queries with different processing strategies."""
    
    def __init__(self, index: Any, processing_mode: str = 'term-at-a-time'):
        """
        Initialize query executor.
        
        Args:
            index: Search index
            processing_mode: 'term-at-a-time' or 'document-at-a-time'
        """
        self.index = index
        self.processing_mode = processing_mode
        self.parser = QueryParser()
    
    def execute(self, query: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
        """
        Execute a Boolean query.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (matching_doc_ids, ranked_results)
        """
        # Parse query
        query_ast = self.parser.parse(query)
        
        # Evaluate to get matching documents
        matching_docs = query_ast.evaluate(self.index)
        
        # Rank results based on index type
        ranked_results = self._rank_results(matching_docs, query)
        
        return matching_docs, ranked_results
    
    def _rank_results(self, doc_ids: Set[str], query: str) -> List[Dict[str, Any]]:
        """
        Rank matching documents.
        
        Args:
            doc_ids: Set of matching document IDs
            query: Original query string
            
        Returns:
            List of ranked results with scores
        """
        # Extract query terms (ignoring operators)
        query_terms = [t.lower() for t in re.findall(r'\b[a-z0-9]+\b', query.lower())]
        
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for doc_id in doc_ids:
            score = 0.0
            
            # Calculate score based on index type
            for term in query_terms:
                if term in self.index.inverted_index:
                    posting = self.index.inverted_index[term].get(doc_id)
                    
                    if posting:
                        if isinstance(posting, list):
                            # Boolean index - count occurrences
                            score += len(posting)
                        elif isinstance(posting, dict):
                            # WordCount or TF-IDF index
                            if 'tfidf' in posting:
                                score += posting['tfidf']
                            elif 'tf' in posting:
                                score += posting['tf']
                            else:
                                score += 1.0
            
            doc_scores[doc_id] = score
        
        # Sort by score (descending)
        ranked = [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return ranked
