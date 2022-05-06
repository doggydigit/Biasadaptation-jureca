from sympy.core.expr import UnevaluatedExpr
import sympy as sp

x, z = sp.symbols('x z')

#expr = (x + 2.*x)/4. + sp.exp((x+sp.UnevaluatedExpr(32.))/6.)
expr = sp.sympify('(x + 2.*x)/4. + exp((x+32.)/6.)', evaluate=False)
expr_ = expr.subs(x, z)

print(expr)
print(expr_)
print('///////////\n')

def convertToRational(expr):
    return sp.nsimplify(sp.parse_expr(sp.ccode(expr), evaluate=False), rational_conversion='exact')

expr_rational = convertToRational(expr)
print('///////////')
print(expr)

substituted = expr_rational.subs(x, z)
print("when converting to Rational, substitution works practically without loss of accuracy:\n", substituted)

wewantthis = expr.subs(x, UnevaluatedExpr(z))
print("we want something like:\n", wewantthis)

print('///////////\n')