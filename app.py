"""Hispanic Heritage Explorer + Function Lab. All application code lives here.

Local: pip install -r requirements.txt && python -m streamlit run app.py
Streamlit Cloud: put app.py and requirements.txt in the GitHub repository root.
Render build: pip install -r requirements.txt
Render start: python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
No API keys, custom modules, external assets, or local data files are required.
"""
from __future__ import annotations

import ast
import hashlib
import io
import math
import re
from datetime import datetime, timezone
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

X = sp.Symbol("x", real=True)
COUNTRIES = {"Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO", "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY"}
# label: (official code, units, description, permitted minimum, permitted maximum)
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected lifespan at birth under prevailing mortality conditions; total population.", 0, None),
    "Unemployment rate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Unemployment as a share of the total labor force; national estimates. Country methods may differ.", 0, 100),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$", "Economic output per person, not household income or personal wealth; not adjusted for inflation.", 0, None),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people", "Birth rate, not the total number of births.", 0, None),
    "Net migration": ("SM.POP.NETM", "people", "Immigration minus emigration for the source measurement period; consult official metadata.", None, None),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "homicides per 100,000 people", "Intentional homicides, as recorded in the source; not every death from violence.", 0, None),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "annual %", "Annual percentage change in consumer prices; negative values represent deflation.", None, None),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central-government debt, not necessarily all public-sector debt.", 0, None),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars.", 0, None),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "Exports minus imports of goods and services; not the current-account balance.", None, None),
}
# Units and variables may be customized in the function lab.
CONTEXTS = {
    "Business": dict(description="A simplified business profit model", independent="time", input_unit="months", dependent="profit", output_unit="thousand dollars", formula="-x^3+12*x^2+20*x+40", interval=(0., 10.), visual="Business / economic indicator"),
    "Engineering": dict(description="Position along a straight track", independent="time", input_unit="seconds", dependent="position", output_unit="meters", formula="x^3-9*x^2+18*x", interval=(0., 8.), visual="Moving object / projectile"),
    "Medical": dict(description="Illustrative drug concentration after a dose", independent="time", input_unit="hours", dependent="drug concentration", output_unit="mg/L", formula="24*x*exp(-0.45*x)", interval=(0., 14.), visual="Tank / concentration"),
    "Sports science": dict(description="Illustrative knee angle during a gait cycle", independent="time", input_unit="seconds", dependent="knee angle", output_unit="degrees", formula="32+24*sin(2*pi*x)+5*sin(4*pi*x)", interval=(0., 2.), visual="Walking person / prosthetic leg"),
    "Biology": dict(description="A logistic population model", independent="time", input_unit="days", dependent="population", output_unit="organisms", formula="800/(1+15*exp(-0.55*x))", interval=(0., 18.), visual="Population / cell colony"),
    "Bioengineering": dict(description="Illustrative prosthetic knee flexion", independent="time", input_unit="seconds", dependent="prosthetic knee angle", output_unit="degrees", formula="34+26*sin(2*pi*x)+4*sin(4*pi*x)", interval=(0., 2.), visual="Walking person / prosthetic leg"),
    "Economic forecasting": dict(description="A hypothetical cyclical economic index", independent="time", input_unit="quarters", dependent="economic index", output_unit="index points", formula="100+15*sin(pi*x/4)+0.8*x", interval=(0., 20.), visual="Business / economic indicator"),
    "Weather": dict(description="A simplified daily temperature cycle", independent="time", input_unit="hours", dependent="temperature", output_unit="degrees C", formula="18+7*sin(pi*(x-8)/12)", interval=(0., 24.), visual="Thermometer / weather"),
    "Custom": dict(description="Describe your model and its assumptions", independent="input", input_unit="input units", dependent="output", output_unit="output units", formula="x^3-6*x^2+9*x+4", interval=(0., 6.), visual="Moving object / projectile"),
}
VISUALS = ["Walking person / prosthetic leg", "Moving object / projectile", "Tank / concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"]


def num(value, digits=5):
    try:
        v = float(value)
        if not np.isfinite(v): return "undefined"
        if v and (abs(v) < 1e-4 or abs(v) >= 1e7): return f"{v:.{digits}g}"
        return f"{v:,.{digits}f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError): return str(value)


def evaluate(expr, values):
    """Real finite values only; vector/scalar expressions return a matching shape."""
    arr = np.asarray(values, dtype=float)
    try:
        fn = sp.lambdify(X, expr, modules="numpy")
        with np.errstate(all="ignore"):
            raw = np.asarray(fn(arr), dtype=complex)
        raw = np.broadcast_to(raw, arr.shape)
        result = np.where((np.abs(raw.imag) < 1e-9) & np.isfinite(raw.real), raw.real, np.nan).copy()
    except Exception:
        result = np.full(arr.shape, np.nan, dtype=float)
    return float(result) if result.ndim == 0 else result


def parse_formula(raw):
    """Whitelist AST parser: never eval/exec user code or use sympify on user text."""
    if len(raw) > 220: raise ValueError("Use a formula of 220 characters or fewer.")
    raw = raw.strip().replace("−", "-").replace("×", "*").replace("÷", "/").replace("^", "**")
    raw = re.sub(r"\b(?:t|year)\b", "x", raw)
    # Familiar implicit multiplication: 2x, 2sin(x), 3(x+1), (x+1)(x-1).
    raw = re.sub(r"(\d|\))\s*(?=[a-zA-Z(])", r"\1*", raw)
    tree = ast.parse(raw, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 90: raise ValueError("This expression is too complex; simplify it.")
    funcs = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan}
    restrictions = []
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not np.isfinite(node.value) or abs(node.value) > 1e12: raise ValueError("Numeric constants must be finite and at most 10^12 in magnitude.")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id in ("x",): return X
            if node.id == "pi": return sp.pi
            if node.id in ("e", "E"): return sp.E
            raise ValueError(f"Unknown name '{node.id}'. Use x (or t), pi, e, and supported functions.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v=walk(node.operand);return v if isinstance(node.op,ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a,b=walk(node.left),walk(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div):
                if b == 0: raise ValueError("Division by zero is not a function.")
                restrictions.append(sp.Ne(b,0));return a/b
            if isinstance(node.op, ast.Pow):
                if b.is_number and (not b.is_real or abs(float(b)) > 12): raise ValueError("Use real constant powers between -12 and 12.")
                if b.is_number and b < 0: restrictions.append(sp.Ne(a,0))
                if b.is_Rational and b.q % 2 == 0:
                    restrictions.append(a >= 0 if b > 0 else a > 0)
                return a**b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs and len(node.args)==1 and not node.keywords:
            arg=walk(node.args[0]); name=node.func.id
            if name in ("log","ln"): restrictions.append(arg>0)
            if name=="sqrt": restrictions.append(arg>=0)
            if name in ("asin","acos"): restrictions.extend([arg>=-1,arg<=1])
            if name=="tan": restrictions.append(sp.Ne(sp.cos(arg),0))
            return funcs[name](arg)
        raise ValueError("Only mathematical arithmetic and supported one-argument functions are allowed. Use * for multiplication if needed.")
    result=walk(tree.body)
    if result.has(sp.zoo,sp.oo,sp.nan): raise ValueError("The formula is not finite.")
    if result.count_ops()>90: raise ValueError("Simplify the formula to reduce complexity.")
    return result, tuple(restrictions)


def validate_table(frame, minimum=None, maximum=None):
    """Reject invalid edits rather than silently changing the working observations."""
    if not {"year","value"}.issubset(frame.columns): raise ValueError("The table needs year and value columns.")
    df=frame[["year","value"]].copy()
    if df.isna().any().any(): raise ValueError("Fill missing year/value entries or remove incomplete rows before fitting.")
    for col in df:
        df[col]=pd.to_numeric(df[col],errors="coerce")
        if not np.isfinite(df[col]).all(): raise ValueError(f"{col} must contain finite numeric values.")
    if not np.allclose(df.year,np.round(df.year),rtol=0,atol=1e-9): raise ValueError("Years must be integers.")
    if not df.year.between(1,9999).all(): raise ValueError("Years must be between 1 and 9999.")
    df.year=df.year.astype(int)
    if df.year.duplicated().any(): raise ValueError("Duplicate years are not permitted; use one observation per year.")
    if minimum is not None and (df.value<minimum).any(): raise ValueError(f"This indicator cannot have values below {minimum}.")
    if maximum is not None and (df.value>maximum).any(): raise ValueError(f"This indicator cannot have values above {maximum}.")
    return df.sort_values("year").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country, code, start, end):
    """Read all returned pages; cache provenance with the response, not today's date."""
    rows=[]; pages=1; page=1; source_updated=""
    url=f"https://api.worldbank.org/v2/country/{country}/indicator/{code}"
    while page<=pages:
        response=requests.get(url,params={"date":f"{start}:{end}","format":"json","per_page":2000,"page":page},timeout=(8,25))
        response.raise_for_status();payload=response.json()
        if not isinstance(payload,list) or len(payload)<2 or not isinstance(payload[0],dict):
            raise ValueError("World Bank returned an unexpected response. Try again or use a verified CSV snapshot.")
        pages=int(payload[0].get("pages",1)); source_updated=payload[0].get("lastupdated","")
        if pages>10: raise ValueError("Unexpectedly large response; narrow the date range.")
        for row in payload[1] or []:
            if row.get("date") and row.get("value") is not None:
                rows.append({"year":int(row["date"]),"value":float(row["value"])})
        page+=1
    df=pd.DataFrame(rows,columns=["year","value"]).sort_values("year").reset_index(drop=True)
    return df, datetime.now(timezone.utc).isoformat(timespec="seconds"), source_updated


def fit_polynomial(df,degree):
    if len(df)<degree+1: raise ValueError(f"Degree {degree} requires at least {degree+1} distinct observed years.")
    base=int(df.year.min());t=df.year.to_numpy(dtype=float)-base;y=df.value.to_numpy(dtype=float)
    # Polynomial.fit scales its design internally; convert back to centered-year units.
    model=np.polynomial.Polynomial.fit(t,y,degree).convert()
    predicted=model(t); residual=y-predicted;denom=np.sum((y-y.mean())**2)
    r2=1-np.sum(residual**2)/denom if denom>1e-20 else np.nan
    metrics={"R-squared (in-sample)":r2,"MAE (in-sample)":float(np.mean(np.abs(residual))),"RMSE (in-sample)":float(np.sqrt(np.mean(residual**2)))}
    expr=sp.Add(*[sp.Float(float(c),16)*X**i for i,c in enumerate(model.coef)])
    return {"base":base,"model":model,"expr":expr,"metrics":metrics,"residual":residual}


def calculation_table(years,fit,df,minimum=None,maximum=None):
    out=[];lo=int(df.year.min());hi=int(df.year.max());observed=dict(zip(df.year,df.value))
    for year in years:
        value=float(fit["model"](year-fit["base"]))
        kind="Fitted value at observed year" if year in observed else "Interpolation" if lo<=year<=hi else "Extrapolation"
        impossible=not np.isfinite(value) or (minimum is not None and value<minimum) or (maximum is not None and value>maximum)
        out.append({"Year":int(year),"Model estimate":value,"Observed value":observed.get(year,np.nan),"Type":kind,"Warning":"Outside valid indicator bounds" if impossible else ""})
    return pd.DataFrame(out)


def parse_years(raw):
    valid=[];errors=[]
    for token in raw.split(","):
        token=token.strip()
        if not token: continue
        if not re.fullmatch(r"\d{4}",token) or not 1950<=int(token)<=2100:
            errors.append(token)
        elif int(token) not in valid: valid.append(int(token))
    return valid,errors


def table_hash(df): return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def bisect_root(expr,a,b):
    fa=evaluate(expr,a)
    for _ in range(50):
        m=(a+b)/2;fm=evaluate(expr,m)
        if not np.isfinite(fm):return None
        if abs(fm)<1e-10:return m
        if fa*fm<=0:b=m
        else:a=m;fa=fm
    root=(a+b)/2
    return root if abs(evaluate(expr,root))<1e-5 else None


def real_roots(expr,lo,hi):
    """Exact polynomial/simple finite solutions; bounded numerical fallback is labeled."""
    if expr==0:return [],"identically zero (every point)"
    found=[];method="numerical candidates"
    try:
        poly=sp.Poly(expr,X)
        if poly.degree()<=12:
            # Exact algebraic roots for exact coefficients; floats use numpy roots.
            if not poly.has(sp.Float):
                roots=sp.polys.polytools.real_roots(poly)
                found=[(float(sp.N(r)),sp.sstr(r)) for r in roots if lo-1e-8<=float(sp.N(r))<=hi+1e-8]
                return list(dict.fromkeys(found)),"exact algebraic roots"
            cs=np.array([float(c) for c in poly.all_coeffs()]);roots=np.roots(cs)
            found=[(float(r.real),None) for r in roots if abs(r.imag)<1e-6 and lo-1e-8<=r.real<=hi+1e-8]
            method="numerical polynomial roots"
            return dedup_roots(found),method
    except Exception:pass
    # Numerator catches rational stationary points, including repeated roots.
    try:
        numerator=sp.together(expr).as_numer_denom()[0]
        if numerator!=expr and sp.Poly(numerator,X).degree()<=12:
            return real_roots(numerator,lo,hi)
    except Exception:pass
    grid=np.linspace(lo,hi,1801);vals=evaluate(expr,grid)
    for i in range(len(grid)-1):
        if np.isfinite(vals[i]) and abs(vals[i])<1e-9:found.append((float(grid[i]),None))
        if np.isfinite(vals[i:i+2]).all() and vals[i]*vals[i+1]<0:
            root=bisect_root(expr,grid[i],grid[i+1])
            if root is not None:found.append((root,None))
    if np.isfinite(vals[-1]) and abs(vals[-1])<1e-9:found.append((float(grid[-1]),None))
    # Approximate even-multiplicity roots: search local minima of |f| without SciPy.
    for i in range(1,len(grid)-1):
        if np.isfinite(vals[i-1:i+2]).all() and abs(vals[i])<abs(vals[i-1]) and abs(vals[i])<abs(vals[i+1]):
            a,b=float(grid[i-1]),float(grid[i+1])
            for _ in range(28):
                m1=a+(b-a)/3;m2=b-(b-a)/3
                if abs(evaluate(expr,m1))<abs(evaluate(expr,m2)):b=m2
                else:a=m1
            root=(a+b)/2
            if np.isfinite(evaluate(expr,root)) and abs(evaluate(expr,root))<1e-7:found.append((root,None))
    return dedup_roots(found),method


def dedup_roots(roots):
    result=[]
    for v,exact in sorted(roots,key=lambda r:r[0]):
        if not result or abs(v-result[-1][0])>1e-5:result.append((v,exact))
    return result


def interval_parts(domain):
    if isinstance(domain,sp.Interval):return [domain]
    if isinstance(domain,sp.Union):return sorted([d for d in domain.args if isinstance(d,sp.Interval)],key=lambda d:float(d.start))
    return []


@st.cache_data(show_spinner=False,max_entries=80)
def analyze_function(expr, restrictions, lo, hi):
    first=sp.diff(expr,X);second=sp.diff(first,X);window=sp.Interval(lo,hi);certified=True;notes=[]
    try:
        domain=sp.calculus.util.continuous_domain(expr,X,window)
        for restriction in restrictions:
            if isinstance(restriction, sp.Unequality):
                forbidden=sp.solveset(restriction.lhs-restriction.rhs,X,domain=window)
                allowed=window-forbidden
            else:
                allowed=sp.solve_univariate_inequality(restriction,X,relational=False)
            domain=domain.intersect(allowed)
        if not interval_parts(domain) and domain!=sp.EmptySet:
            certified=False;notes.append("The domain could not be reduced to intervals; use the numerical plot and verify restrictions independently.")
    except Exception:
        domain=window;certified=False;notes.append("Domain analysis is numerical/incomplete for this expression; undefined samples are omitted.")
    parts=interval_parts(domain)
    if not parts and domain==sp.EmptySet:
        return dict(first=first,second=second,domain=domain,certified=certified,parts=[],critical=[],inflections=[],trend=[],concavity=[],absolute=[],breaks=[],notes=["No real continuous interval exists in this viewing window."],root_method="none")
    if not parts:parts=[window]
    breaks=[]
    for part in parts:
        if lo<float(part.start)<hi:breaks.append(float(part.start))
        if lo<float(part.end)<hi:breaks.append(float(part.end))
    # Include isolated nondifferentiable points (sqrt, abs, inverse trig endpoints).
    derivative_breaks=[]
    try:
        d_domain=sp.calculus.util.continuous_domain(first,X,window)
        for part in interval_parts(d_domain):
            for v in (part.start,part.end):
                if lo<float(v)<hi and np.isfinite(evaluate(expr,float(v))):derivative_breaks.append(float(v))
    except Exception:
        if expr.has(sp.Abs):
            for atom in expr.atoms(sp.Abs):
                derivative_breaks += [v for v,_ in real_roots(atom.args[0],lo,hi)[0]]
    roots,method=real_roots(first,lo,hi);candidates=dedup_roots(roots+[(v,None) for v in derivative_breaks])
    critical=[];trend=[];concavity=[];inflections=[]
    for part in parts:
        left,right=float(part.start),float(part.end)
        inside=[(r,exact) for r,exact in candidates if left<r<right and np.isfinite(evaluate(expr,r))]
        boundaries=sorted(set([left,right]+[r for r,_ in inside]))
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            s=evaluate(first,(a+b)/2)
            trend.append((a,b,"Increasing" if s>1e-8 else "Decreasing" if s< -1e-8 else "Constant / numerically flat"))
        for r,exact in inside:
            idx=boundaries.index(r);delta=min(r-boundaries[idx-1],boundaries[idx+1]-r)*.02
            s1=evaluate(first,r-delta);s2=evaluate(first,r+delta)
            if s1>0 and s2<0:kind="Local maximum";reason="f' changes + to -"
            elif s1<0 and s2>0:kind="Local minimum";reason="f' changes - to +"
            else:kind="Critical point; no sign-change extremum";reason="f' does not change sign, or test is inconclusive"
            if r in derivative_breaks:reason+="; derivative is undefined here"
            critical.append(dict(x=r,y=evaluate(expr,r),exact=exact,kind=kind,reason=reason))
        secroots,_=real_roots(second,left,right)
        cands=dedup_roots(secroots+[(v,None) for v in derivative_breaks])
        cb=sorted(set([left,right]+[v for v,_ in cands if left<v<right]))
        for a,b in zip(cb[:-1],cb[1:]):
            value=evaluate(second,(a+b)/2)
            concavity.append((a,b,"Concave up" if value>1e-8 else "Concave down" if value< -1e-8 else "Zero / undefined curvature"))
        for r,exact in cands:
            if not left<r<right or not np.isfinite(evaluate(expr,r)):continue
            i=cb.index(r);d=min(r-cb[i-1],cb[i+1]-r)*.02;s1=evaluate(second,r-d);s2=evaluate(second,r+d)
            if np.isfinite(s1) and np.isfinite(s2) and s1*s2<0:
                inflections.append(dict(x=r,y=evaluate(expr,r),exact=exact,kind="Inflection (concavity changes)",reason="f'' changes sign"))
    absolute=[]
    # Only claim interval absolute extrema when continuity over the closed interval is certified.
    if certified and domain==window:
        points=[lo,hi]+[c["x"] for c in critical];values=[evaluate(expr,v) for v in points]
        if np.isfinite(values).all():
            mn=min(values);mx=max(values)
            absolute=[dict(kind="Absolute minimum",x=v,y=y) for v,y in zip(points,values) if np.isclose(y,mn,rtol=1e-8,atol=1e-8)]+[dict(kind="Absolute maximum",x=v,y=y) for v,y in zip(points,values) if np.isclose(y,mx,rtol=1e-8,atol=1e-8)]
            if method.startswith("numerical"):notes.append("Extrema are numerical candidates, not a proof that all critical points were found.")
    else:notes.append("Absolute extrema are not asserted: continuity on the entire selected closed interval is not certified. Analyze each valid component separately.")
    if expr==sp.simplify(expr) and not expr.has(X):notes.append("Constant function: every point attains the same maximum/minimum; the derivative is identically zero.")
    return dict(first=first,second=second,domain=domain,certified=certified,parts=parts,critical=critical,inflections=inflections,trend=trend,concavity=concavity,absolute=absolute,breaks=sorted(set(breaks)),notes=notes,root_method=method)


def domain_contains(domain,v):
    try:return bool(domain.contains(sp.Float(v)))
    except Exception:return True


def graph_samples(expr,analysis,lo,hi):
    """Sample each valid component separately so poles/holes are never joined."""
    xx=[];yy=[]
    for part in analysis["parts"]:
        a,b=float(part.start),float(part.end);eps=max((b-a)*1e-7,1e-9)
        a+=eps if part.left_open else 0;b-=eps if part.right_open else 0
        if b<=a:continue
        xs=np.linspace(a,b,500);ys=evaluate(expr,xs)
        # Retain domain restrictions on canceled expressions too.
        if not analysis["certified"]:
            for i,z in enumerate(xs):
                if not domain_contains(analysis["domain"],z):ys[i]=np.nan
        xx.extend(xs.tolist()+[np.nan]);yy.extend(ys.tolist()+[np.nan])
    return np.array(xx),np.array(yy)


def derivative_samples(expr,analysis,lo,hi):
    local=analysis.copy()
    try:
        domain=sp.calculus.util.continuous_domain(expr,X,sp.Interval(lo,hi)).intersect(analysis["domain"])
        local["domain"]=domain;local["parts"]=interval_parts(domain)
    except Exception:
        pass
    return graph_samples(expr,local,lo,hi)


def simulation(visual,value,bounds,phase,meta,config):
    low,high=bounds;state=float(np.clip((value-low)/(high-low),0,1)) if np.isfinite(value) and high>low else 0.
    fig=go.Figure();fig.update_layout(height=350,margin=dict(l=10,r=10,t=45,b=15),title=visual,paper_bgcolor="#f5f8fb",plot_bgcolor="#f5f8fb",showlegend=False)
    fig.update_xaxes(range=[-1,1],visible=False);fig.update_yaxes(range=[-.1,1.8],visible=False,scaleanchor="x")
    if not np.isfinite(value):
        fig.add_annotation(x=0,y=.8,text="No real value at this input",showarrow=False);return fig
    if visual==VISUALS[0]:
        height=config.get("height",1.7);thigh=.245*height;shank=.246*height
        stride=config.get("stride",.7);hipx=-.55+1.1*phase;hipy=thigh+shank+.12;shoulder=hipy+.35*height
        sway=math.radians(10+20*stride)*math.sin(2*math.pi*phase)
        knees=[]
        for side in (0,1):
            angle=sway if side else -sway
            prosthetic=(side==1) if config.get("side","Right")=="Right" else (side==0)
            flex=math.radians(5+65*state) if prosthetic else .15*max(0,-math.sin(2*math.pi*phase))
            knee=(hipx+thigh*math.sin(angle),hipy-thigh*math.cos(angle))
            foot=(knee[0]+shank*math.sin(angle-flex),knee[1]-shank*math.cos(angle-flex))
            fig.add_trace(go.Scatter(x=[hipx,knee[0],foot[0]],y=[hipy,knee[1],foot[1]],mode="lines+markers",line=dict(width=8,color="#e98925" if prosthetic else "#227bb5"),marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=[hipx,hipx],y=[hipy,shoulder],mode="lines",line=dict(width=10,color="#334155")))
        fig.add_trace(go.Scatter(x=[hipx],y=[shoulder+.13*height],mode="markers",marker=dict(size=26,color="#cf9d72")))
        fig.add_shape(type="line",x0=-1,x1=1,y0=0,y1=0,line_color="#64748b")
    elif visual==VISUALS[1]:
        horizontal=config.get("direction","Horizontal")=="Horizontal"
        cx,cy=(-.8+1.6*state,.5) if horizontal else (0,.15+1.2*state)
        fig.add_shape(type="circle",x0=cx-.12,x1=cx+.12,y0=cy-.12,y1=cy+.12,fillcolor="#007e83",line_color="#173447")
        fig.add_shape(type="line",x0=-.95,x1=.95,y0=.05,y1=.05,line_color="#64748b")
    elif visual==VISUALS[2]:
        capacity=config.get("capacity",100.);fill=float(np.clip(value/capacity,0,1))
        fig.add_shape(type="rect",x0=-.35,x1=.35,y0=.05,y1=1.5,line_color="#173447",fillcolor="white")
        fig.add_shape(type="rect",x0=-.32,x1=.32,y0=.08,y1=.08+1.38*fill,fillcolor="#007e83",line_width=0)
    elif visual==VISUALS[3]:
        maximum=config.get("icons",36);n=max(0,int(round(state*maximum)))
        fig.add_trace(go.Scatter(x=[-.75+.3*(i%6) for i in range(n)],y=[1.4-.22*(i//6) for i in range(n)],mode="markers",marker=dict(size=18,color="#007e83")))
    elif visual==VISUALS[4]:
        fig.add_shape(type="rect",x0=-.08,x1=.08,y0=.2,y1=1.45,fillcolor="white",line_color="#173447")
        fig.add_shape(type="rect",x0=-.05,x1=.05,y0=.2,y1=.2+1.2*state,fillcolor="#ef4444",line_width=0)
        fig.add_shape(type="circle",x0=-.18,x1=.18,y0=0,y1=.36,fillcolor="#ef4444",line_color="#173447")
    else:
        fig.add_trace(go.Bar(x=[0],y=[1.4*state],width=.7,marker_color="#007e83"))
    fig.add_annotation(x=0,y=1.65,text=f"{escape(meta['dependent'])}: {num(value)} {escape(meta['output_unit'])}",showarrow=False,font_size=12)
    return fig


def plot_static(df,fit,category,unit,horizon,results,comparisons):
    fig,ax=plt.subplots(figsize=(9,4.5));years=df.year.to_numpy();hi=int(years.max())
    ax.scatter(years,df.value,s=14,label="Working observations",color="#147d92")
    xs=np.linspace(years.min(),hi,400);ax.plot(xs,fit["model"](xs-fit["base"]),label="Polynomial fit",color="#173447")
    if horizon:
        xs=np.linspace(hi,hi+horizon,200);ax.plot(xs,fit["model"](xs-fit["base"]),"--",label="Extrapolated",color="#e98925")
    for name,frame in comparisons.items():ax.plot(frame.year,frame.value,"o-",markersize=2,label=name,alpha=.7)
    if not results.empty:ax.scatter(results.Year,results["Model estimate"],marker="D",label="Requested estimates",color="#9333a8")
    ax.set(xlabel="Year",ylabel=unit,title=category);ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format="png",dpi=140);plt.close(fig);return buf.getvalue()


def export_excel(metadata,df,results,residuals,samples,comparisons):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="xlsxwriter",engine_kwargs={"options":{"strings_to_formulas":False,"strings_to_urls":False}}) as writer:
        pd.DataFrame(metadata.items(),columns=["Field","Value"]).to_excel(writer,sheet_name="Summary",index=False)
        df.to_excel(writer,sheet_name="Working data",index=False)
        residuals.to_excel(writer,sheet_name="Residuals",index=False)
        samples.to_excel(writer,sheet_name="Sampled model",index=False)
        if not results.empty:results.to_excel(writer,sheet_name="Calculated years",index=False)
        for name,frame in comparisons.items():frame.to_excel(writer,sheet_name=name[:31],index=False)
        for sheet in writer.sheets.values():sheet.set_column(0,0,28);sheet.set_column(1,5,26)
    return buf.getvalue()


def export_pdf(metadata,df,results,chart,findings,title="Hispanic Heritage Historical Data Explorer"):
    buf=io.BytesIO();styles=getSampleStyleSheet();story=[]
    story.append(Paragraph(escape(title),styles["Title"]))
    for key,value in metadata.items():
        story.append(Paragraph(f"<b>{escape(str(key))}:</b> {escape(str(value))}",styles["BodyText"]))
    story.append(Spacer(1,12));story.append(Image(io.BytesIO(chart),width=500,height=250))
    story.append(Paragraph("Findings and limitations",styles["Heading2"]))
    for finding in findings:story.append(Paragraph(escape(str(finding)),styles["BodyText"]))
    def add_table(title,frame):
        story.append(Paragraph(title,styles["Heading2"]))
        data=[[Paragraph(escape(str(c)),styles["BodyText"]) for c in frame.columns]]
        for row in frame.itertuples(index=False,name=None):
            data.append([Paragraph(escape(num(v) if isinstance(v,(int,float,np.number)) else str(v)),styles["BodyText"]) for v in row])
        widths=[500/len(frame.columns)]*len(frame.columns)
        table=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceef0")),("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.3,colors.lightgrey),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)]))
        story.append(table)
    story.append(PageBreak());add_table("Working data preview (first 20 observations)",df.head(20))
    if not results.empty:add_table("Requested year calculations",results)
    doc=SimpleDocTemplate(buf,pagesize=letter,leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=42)
    doc.build(story);return buf.getvalue()


def data_explorer():
    st.header("Historical data explorer")
    st.caption("Authentic World Bank data • Editable scenarios • Polynomial modeling • Portable reports")
    controls=st.columns([1,1,1])
    country=controls[0].selectbox("Country",list(COUNTRIES),key="country")
    category=controls[1].selectbox("Indicator",list(INDICATORS),key="indicator")
    degree=controls[2].slider("Polynomial degree",3,8,3,key="degree")
    code,unit,definition,minimum,maximum=INDICATORS[category]
    now=datetime.now(timezone.utc);start,end=now.year-70,now.year-1
    st.write(definition);st.markdown(f"[Official indicator and metadata](https://data.worldbank.org/indicator/{code}) · **Units:** {unit} · Search window: {start}–{end}")
    st.caption("Brazil is included as a Latin American comparison; it is not a Spanish-speaking Hispanic country. National unemployment definitions can differ. Missing observations are not invented.")
    source_key=(country,code,start,end)
    if st.button("Refresh World Bank data"):
        fetch_world_bank.clear();st.session_state.pop("loaded_source",None);st.rerun()
    status=st.empty()
    if st.session_state.get("loaded_source")!=source_key:
        try:
            with st.spinner("Retrieving official observations…"):
                source,retrieved,updated=fetch_world_bank(COUNTRIES[country],code,start,end)
            st.session_state.source=source;st.session_state.retrieved=retrieved;st.session_state.updated=updated
            st.session_state.loaded_source=source_key;st.session_state.source_error=""
        except Exception as exc:
            st.session_state.source=pd.DataFrame(columns=["year","value"]);st.session_state.retrieved="";st.session_state.updated=""
            st.session_state.loaded_source=source_key;st.session_state.source_error=str(exc)
    source=st.session_state.source
    if st.session_state.get("source_error"):status.warning(f"Live retrieval unavailable: {st.session_state.source_error}")
    if source.empty:st.warning("No official observations are available for this selection. Choose another indicator/country or upload a verified snapshot below.")
    with st.expander("Use a verified CSV snapshot / inspect original data",expanded=source.empty):
        st.write("Optional CSV columns: year,value. It is your supplied dataset, not a new live API response. Verify the selected indicator, country, units, and source before uploading.")
        upload=st.file_uploader("Verified historical CSV",type="csv",key=f"upload_{country}_{code}")
        upload_note=st.text_input("Uploaded data source URL / provenance",key=f"provenance_{country}_{code}")
        st.dataframe(source,width="stretch")
    working_source=source;source_note=f"World Bank: https://data.worldbank.org/indicator/{code}";stamp=st.session_state.retrieved
    if upload is not None:
        if not upload_note.strip():st.info("Add provenance for the uploaded CSV before analysis.");return
        try:working_source=validate_table(pd.read_csv(upload),minimum,maximum)
        except Exception as exc:st.error(f"CSV rejected: {exc}");return
        source_note=f"User-provided CSV; claimed source: {upload_note}";stamp="User-uploaded snapshot (not retrieved by this app)"
    if working_source.empty:return
    st.caption(f"Available observations: {len(working_source)} · {int(working_source.year.min())}–{int(working_source.year.max())} · Retrieved: {stamp} · Source update: {st.session_state.updated or 'not provided'}")
    origin_hash=hashlib.sha256((str(source_key)+table_hash(working_source)).encode()).hexdigest()[:16]
    reset_key=f"reset_{origin_hash}"
    if reset_key not in st.session_state:st.session_state[reset_key]=0
    if st.button("Reset working table to source"):
        st.session_state[reset_key]+=1;st.rerun()
    edited=st.data_editor(working_source,num_rows="dynamic",hide_index=True,width="stretch",key=f"editor_{origin_hash}_{st.session_state[reset_key]}",column_config={"year":st.column_config.NumberColumn("Year",format="%d"),"value":st.column_config.NumberColumn(f"Value ({unit})",format="%.6f")})
    try:df=validate_table(edited,minimum,maximum);fit=fit_polynomial(df,degree)
    except ValueError as exc:st.error(str(exc));return
    modified=not df.equals(working_source)
    if modified:st.warning("User-modified scenario: these edits are not official observations. All estimates and reports use this working table.")
    fingerprint=hashlib.sha256((str(source_key)+source_note+str(degree)+table_hash(df)).encode()).hexdigest()
    if st.session_state.get("prediction_fingerprint")!=fingerprint:
        st.session_state.prediction_fingerprint=fingerprint;st.session_state.year_results={};st.session_state.batch_results=pd.DataFrame()
    options=st.columns(3)
    increment=options[0].slider("Model sample increment (years)",1,10,1)
    horizon=options[1].slider("Forward display horizon (years)",0,50,5)
    printer=options[2].checkbox("Printer-friendly report view")
    selected=st.multiselect("Compare same indicator across countries",[c for c in COUNTRIES if c!=country])
    comparisons={}
    for other in selected:
        try:
            other_df,_,_=fetch_world_bank(COUNTRIES[other],code,start,end)
            if other_df.empty:st.info(f"{other}: no observations for this indicator.")
            else:comparisons[other]=other_df
        except Exception as exc:st.warning(f"{other}: retrieval unavailable ({exc}).")
    samples_year=np.arange(int(df.year.min()),int(df.year.max())+horizon+1,increment)
    samples=calculation_table(samples_year,fit,df,minimum,maximum)
    st.latex("t=\\mathrm{year}-"+str(fit["base"])+",\\qquad f(t)="+sp.latex(fit["expr"].subs(X,sp.Symbol("t"))))
    metric_cols=st.columns(4)
    metric_cols[0].metric("R² · in-sample",num(fit["metrics"]["R-squared (in-sample)"]))
    metric_cols[1].metric(f"MAE · {unit}",num(fit["metrics"]["MAE (in-sample)"]))
    metric_cols[2].metric("Observations",len(df));metric_cols[3].metric(f"Mean · {unit}",num(df.value.mean()))
    single,batch=st.columns(2)
    with single:
        st.subheader("Single-year calculation")
        year=st.number_input("Requested year",1950,2100,min(max(now.year,1950),2100),key="single_year")
        if st.button("Calculate Value"):
            result=calculation_table([year],fit,df,minimum,maximum);st.session_state.year_results=result.to_dict("records")[0]
        result=st.session_state.year_results
        if result:st.dataframe(pd.DataFrame([result]),hide_index=True,width="stretch")
    with batch:
        st.subheader("Batch-year calculation")
        mode=st.radio("Year selection",["Year range","Custom years"],horizontal=True)
        valid=[]
        if mode=="Year range":
            a,b,c=st.columns(3);y0=a.number_input("Start year",1950,2100,2025);y1=b.number_input("End year",1950,2100,2030);step=c.number_input("Step",1,10,1)
            if y1<=y0:st.warning("End year must exceed start year.")
            else:valid=list(range(int(y0),int(y1)+1,int(step)))
        else:
            raw=st.text_input("Comma-separated years","2025, 2030, 2035")
            valid,invalid=parse_years(raw)
            if invalid:st.warning("Rejected entries: "+", ".join(invalid))
        st.caption("Requested years: "+(", ".join(map(str,valid)) if valid else "none"))
        if st.button("Calculate Batch",disabled=not valid):st.session_state.batch_results=calculation_table(valid,fit,df,minimum,maximum)
        if not st.session_state.batch_results.empty:st.dataframe(st.session_state.batch_results,hide_index=True,width="stretch")
    frames=[]
    if st.session_state.year_results:frames.append(pd.DataFrame([st.session_state.year_results]))
    if not st.session_state.batch_results.empty:frames.append(st.session_state.batch_results)
    results=pd.concat(frames,ignore_index=True).drop_duplicates("Year") if frames else pd.DataFrame()
    if (not results.empty and results.Warning.ne("").any()) or samples.Warning.ne("").any():st.warning("Some polynomial estimates violate this indicator's valid bounds. They are shown without clipping; extrapolation is not a reliable forecast.")
    chart=plot_static(df,fit,f"{category} — {country}",unit,horizon,results,comparisons)
    if printer:st.image(chart,caption="Printer-friendly historical/model report")
    else:
        fig=go.Figure();fig.add_trace(go.Scatter(x=df.year,y=df.value,mode="markers",name=country+(" · edited scenario" if modified else " · source")))
        years=np.linspace(df.year.min(),df.year.max(),450);fig.add_trace(go.Scatter(x=years,y=fit["model"](years-fit["base"]),name="Polynomial fit",line_color="#173447"))
        if horizon:
            years=np.linspace(df.year.max(),df.year.max()+horizon,180);fig.add_trace(go.Scatter(x=years,y=fit["model"](years-fit["base"]),name="Extrapolation",line=dict(dash="dash",color="#e98925")))
        fig.add_trace(go.Scatter(x=samples.Year,y=samples["Model estimate"],mode="markers",name=f"Model samples · every {increment} years",marker=dict(symbol="circle-open",size=7)))
        for name,frame in comparisons.items():fig.add_trace(go.Scatter(x=frame.year,y=frame.value,mode="markers",name=name))
        if not results.empty:
            for extrapolate in (False,True):
                subset=results[results.Type.eq("Extrapolation")==extrapolate]
                if not subset.empty:fig.add_trace(go.Scatter(x=subset.Year,y=subset["Model estimate"],mode="markers",name="Requested extrapolations" if extrapolate else "Requested within-range estimates",marker=dict(symbol="triangle-up" if extrapolate else "diamond",size=12)))
        fig.add_vline(x=float(df.year.max()),line_dash="dot")
        fig.update_layout(template="plotly_white",height=500,xaxis_title="Year",yaxis_title=f"{category} ({unit})",legend=dict(orientation="h"))
        st.plotly_chart(fig,width="stretch")
    st.caption("Comparisons show source observations, not independently fitted forecast curves. Gaps and country coverage may differ; no missing values are filled.")
    residuals=df.copy();residuals["fitted"]=fit["model"](df.year.to_numpy()-fit["base"]);residuals["residual (observed minus fitted)"]=fit["residual"]
    with st.expander("Model diagnostics, residuals, and degree comparison"):
        residual_fig=go.Figure(go.Scatter(x=df.year,y=fit["residual"],mode="markers",name="Residual"));residual_fig.add_hline(y=0);residual_fig.update_layout(xaxis_title="Year",yaxis_title=f"Residual ({unit})",height=280);st.plotly_chart(residual_fig,width="stretch")
        other_degree=st.selectbox("Compare a second degree",[d for d in range(3,9) if d!=degree])
        rows=[]
        for d in (degree,other_degree):
            try:ff=fit_polynomial(df,d);rows.append({"Degree":d,**ff["metrics"]})
            except ValueError:pass
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        holdout=max(3,int(math.ceil(len(df)*.2)))
        if len(df)-holdout>=degree+1:
            train=df.iloc[:-holdout];test=df.iloc[-holdout:];train_fit=fit_polynomial(train,degree);pred=train_fit["model"](test.year.to_numpy()-train_fit["base"])
            st.write(f"Chronological holdout: fit {len(train)} earlier observations; evaluate {len(test)} later observations. Holdout MAE: **{num(np.mean(np.abs(test.value.to_numpy()-pred)))} {unit}**. This is separate from the full-series fit.")
        else:st.info("Not enough observations for a chronological holdout of at least three years with this degree.")
        st.dataframe(residuals,hide_index=True);st.dataframe(samples,hide_index=True)
    observed_change=float(df.value.iloc[-1]-df.value.iloc[0]);annual=observed_change/(df.year.iloc[-1]-df.year.iloc[0])
    rate_unit="percentage points per year" if "%" in unit else f"{unit} per year"
    findings=[f"Observed working-series change from {df.year.iloc[0]} to {df.year.iloc[-1]}: {num(observed_change)} {unit}.",f"Average observed change: {num(annual)} {rate_unit}.","In-sample fit is not forecast validation. High-degree polynomial extrapolations can become implausible.","MAE and residuals describe fit error; statistically validated prediction intervals are not provided.","Indicators alone cannot establish causes or represent every dimension of a community's experience."]
    st.subheader("Findings and heritage connection")
    for text in findings:st.write(text)
    question=st.text_input("Research question",key="research_question")
    heritage=st.text_area("Your sourced heritage/history connection (include source URL)",key="heritage_connection")
    if question:findings.insert(0,"Research question: "+question)
    if heritage:findings.append("Student-supplied heritage connection: "+heritage)
    metadata={"Country":country,"Indicator":category,"Units":unit,"Source":source_note,"Retrieved":stamp,"Source update":st.session_state.updated,"Generated UTC":now.isoformat(timespec="seconds"),"Data status":"User-modified scenario" if modified else "Uploaded verified snapshot" if upload is not None else "Source observations","Coverage":f"{int(df.year.min())}–{int(df.year.max())}","Count":len(df),"Degree":degree,"Base year":fit["base"],"Equation in t = year - base":sp.sstr(fit["expr"].subs(X,sp.Symbol("t"))),**{k:num(v) for k,v in fit["metrics"].items()},"Mean":num(df.value.mean()),"Comparison countries":", ".join(comparisons)}
    st.session_state.fitted_bridge=dict(expr=fit["expr"],base=fit["base"],lo=0.,hi=float(df.year.max()-fit["base"]+horizon),country=country,category=category,unit=unit,status=metadata["Data status"],fingerprint=fingerprint)
    st.subheader("Download current report")
    with st.spinner("Preparing PDF and Excel reports…"):
        pdf=export_pdf(metadata,df,results,chart,findings);excel=export_excel(metadata,df,results,residuals,samples,comparisons)
    cols=st.columns(3);name=re.sub(r"[^A-Za-z0-9]+","_",f"{country}_{category}")
    cols[0].download_button("PDF report",pdf,f"{name}.pdf","application/pdf")
    cols[1].download_button("Excel workbook",excel,f"{name}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    cols[2].download_button("CSV working data",df.to_csv(index=False).encode(),f"{name}.csv","text/csv")
    st.caption("CSV contains the current working table. Its provenance and edit status are recorded in the PDF/Excel summary. Export files are generated in memory; no local files are needed.")
    st.success("This fitted polynomial is now available in the Function Lab: choose 'Retrieved-data polynomial'.")


def build_parameter_model(context):
    """Parameters control the actual function; visual settings are separate."""
    if context in ("Business","Economic forecasting","Weather","Custom"):
        baseline=st.number_input("Baseline",value=100. if context!="Weather" else 18.,key="par_base")
        trend=st.number_input("Linear trend per input unit",value=1.,key="par_trend")
        amplitude=st.number_input("Cycle amplitude",min_value=0.,value=15.,key="par_amp")
        period=st.number_input("Cycle period (input units)",min_value=.1,value=8.,key="par_period")
        return sp.Float(baseline)+sp.Float(trend)*X+sp.Float(amplitude)*sp.sin(2*sp.pi*X/sp.Float(period))
    if context=="Engineering":
        initial=st.number_input("Initial position",value=2.,key="par_init")
        velocity=st.number_input("Initial velocity",value=24.,key="par_v")
        acceleration=st.number_input("Constant acceleration",value=-9.8,key="par_a")
        return sp.Float(initial)+sp.Float(velocity)*X+sp.Float(acceleration)*X**2/2
    if context=="Medical":
        scale=st.number_input("Dose-response scale",min_value=.01,value=24.,key="par_dose")
        elimination=st.number_input("Elimination constant",min_value=.001,value=.45,key="par_elim")
        return sp.Float(scale)*X*sp.exp(-sp.Float(elimination)*X)
    if context=="Biology":
        capacity=st.number_input("Carrying capacity",min_value=1.,value=800.,key="par_k")
        population=st.number_input("Initial population",min_value=.1,value=50.,key="par_pop")
        growth=st.number_input("Growth-rate constant",min_value=.001,value=.55,key="par_growth")
        return sp.Float(capacity)/(1+sp.Float(capacity/population-1)*sp.exp(-sp.Float(growth)*X))
    center=st.number_input("Center angle (degrees)",value=32.,key="par_angle")
    amplitude=st.number_input("Flexion amplitude (degrees)",min_value=0.,value=24.,key="par_flex")
    frequency=st.number_input("Cycles per second",min_value=.01,value=1.,key="par_freq")
    return sp.Float(center)+sp.Float(amplitude)*sp.sin(2*sp.pi*sp.Float(frequency)*X)


def features_frame(items):
    return pd.DataFrame([{ "x ≈":r["x"],"f(x) ≈":r["y"],"Exact x (if available)":r.get("exact") or "numerical", "Feature":r["kind"],"Justification":r.get("reason","")} for r in items])


def contextual_text(meta,x,y,slope,curvature):
    if not np.isfinite(y):return f"At {meta['independent']} = {num(x)} {meta['input_unit']}, this formula has no real finite value in its permitted domain."
    first_unit=f"{meta['output_unit']} per {meta['input_unit']}";second_unit=f"{meta['output_unit']} per {meta['input_unit']} squared"
    direction="increasing" if slope>1e-8 else "decreasing" if slope< -1e-8 else "momentarily flat" if np.isfinite(slope) else "with an undefined derivative"
    if np.isfinite(curvature):
        change="the signed rate of change is increasing" if curvature>1e-8 else "the signed rate of change is decreasing" if curvature< -1e-8 else "the second derivative is zero at this point"
    else:change="the second derivative is undefined here"
    text=f"At **{meta['independent']} = {num(x)} {meta['input_unit']}**, **{meta['dependent']} = {num(y)} {meta['output_unit']}**. It is {direction}. Its instantaneous rate is **{num(slope)} {first_unit}**; {change} (**{num(curvature)} {second_unit}**)."
    if meta["dependent"].lower() in ("position","displacement"):
        text+=" Here f′ is velocity and f″ is acceleration. Negative acceleration means velocity is decreasing; it does not always mean speed is decreasing."
    elif slope<0 and curvature>0:text+=" The quantity is falling, but its decline is slowing."
    elif slope<0 and curvature<0:text+=" The quantity is falling at an increasing magnitude of decline."
    return text


def function_lab():
    st.header("Function analysis lab")
    st.caption("Choose a context, explore a model, and connect graphs, derivatives, and motion.")
    setup=st.columns(3)
    level=setup[0].selectbox("Course level",["Honors Algebra 2","Precalculus","Calculus"],index=2)
    context=setup[1].selectbox("Context",list(CONTEXTS))
    sources=["Suggested function","Build with model parameters","Enter my own function","Retrieved-data polynomial"]
    source=setup[2].selectbox("Function source",sources)
    ctx=CONTEXTS[context].copy(); restrictions=();bridge=st.session_state.get("fitted_bridge")
    if source=="Retrieved-data polynomial":
        if bridge is None:st.info("First open Historical data, fit a real series, then return here. You can use the other function sources immediately.");return
        expr=bridge["expr"];ctx.update(description=f"{bridge['category']} in {bridge['country']} ({bridge['status']})",independent=f"years after {bridge['base']}",input_unit="years",dependent=bridge["category"],output_unit=bridge["unit"],interval=(bridge["lo"],bridge["hi"]),visual=VISUALS[5])
        st.info(f"Fitted data model: t = calendar year − {bridge['base']}. Values beyond the observed range are extrapolations. The model snapshot updates after you refit the historical table.")
    elif source=="Build with model parameters":
        with st.expander("Parameters shaping the function",expanded=True):expr=build_parameter_model(context)
    else:
        raw=st.text_input("Function f(x) (x or t)",ctx["formula"],key=f"formula_{context}_{source}",help="Examples: x^3-3x, 1/(x-2), exp(-x), log(x), sqrt(x), sin(2*pi*x). Supports 2x; use * when notation is ambiguous.")
        try:expr,restrictions=parse_formula(raw)
        except Exception as exc:st.error(f"Formula rejected: {exc}");return
    with st.expander("Context, variable names, units, and analysis interval",expanded=True):
        desc=st.text_input("Context description",ctx["description"],key=f"desc_{context}_{source}")
        c=st.columns(4)
        independent=c[0].text_input("Independent quantity",ctx["independent"],key=f"indep_{context}_{source}")
        input_unit=c[1].text_input("Input units",ctx["input_unit"],key=f"inunit_{context}_{source}")
        dependent=c[2].text_input("Dependent quantity",ctx["dependent"],key=f"dep_{context}_{source}")
        output_unit=c[3].text_input("Output units",ctx["output_unit"],key=f"outunit_{context}_{source}")
        c=st.columns(2);lo=c[0].number_input("Analysis interval start",value=ctx["interval"][0],key=f"lo_{context}_{source}");hi=c[1].number_input("Analysis interval end",value=ctx["interval"][1],key=f"hi_{context}_{source}")
    if not np.isfinite([lo,hi]).all() or hi<=lo or hi-lo>10000:st.error("Use a finite interval with end greater than start and length at most 10,000.");return
    meta=dict(description=desc,independent=independent,input_unit=input_unit,dependent=dependent,output_unit=output_unit)
    try:
        with st.spinner("Analyzing derivatives, domain, and graph features…"):analysis=analyze_function(expr,restrictions,float(lo),float(hi))
    except Exception as exc:st.error(f"This expression could not be analyzed: {exc}. Try a simpler formula or narrower interval.");return
    if not analysis["parts"]:st.warning("No real continuous part of this function exists on the selected interval.");return
    st.latex("f(x)="+sp.latex(expr))
    if level=="Calculus":
        st.latex("f'(x)="+sp.latex(analysis["first"]));st.latex("f''(x)="+sp.latex(analysis["second"]))
    st.write("**Real continuous domain within this window:**");st.latex(sp.latex(analysis["domain"]))
    for restriction in restrictions:st.caption("Original-expression restriction: "+sp.sstr(restriction))
    st.caption("Exact domain/roots are labeled where available. Decimals, plotted curves, and data fits are approximations. Numerical searches can miss closely spaced or highly oscillatory features; verify them independently.")
    for note in analysis["notes"]:st.info(note)
    c=st.columns(5)
    show_function=c[0].checkbox("Show f",True)
    show_first=c[1].checkbox("Show f′",False,disabled=level!="Calculus") and level=="Calculus"
    show_second=c[2].checkbox("Show f″",False,disabled=level!="Calculus") and level=="Calculus"
    show_secant=c[3].checkbox("Show secant",True)
    show_tangent=c[4].checkbox("Show tangent",True,disabled=level!="Calculus") and level=="Calculus"
    visual=st.selectbox("Visual simulation",VISUALS,index=VISUALS.index(ctx["visual"]),key=f"visual_{context}_{source}")
    config={}
    with st.expander("Visual scaling and simulation settings"):
        if visual==VISUALS[0]:
            c=st.columns(3);config["height"]=c[0].slider("Person height (m)",1.2,2.1,1.7,.01);config["stride"]=c[1].slider("Stride length (m)",.25,1.4,.7,.01);config["side"]=c[2].selectbox("Prosthetic leg",["Right","Left"])
            st.write("Input drives gait phase and horizontal travel; normalized f(x) drives prosthetic knee flexion from 5° to 70°. Stride changes the leg swing. This is a conceptual walker, not a biomechanical validation.")
        elif visual==VISUALS[1]:config["direction"]=st.radio("Motion direction",["Horizontal","Vertical"],horizontal=True)
        elif visual==VISUALS[2]:
            config["capacity"]=st.number_input("Visual reference capacity (output units)",min_value=.001,value=100.)
            st.write("Tank fill = f(x)/reference capacity, clipped for drawing only. Exact values remain shown, including negatives or values above capacity.")
        elif visual==VISUALS[3]:config["icons"]=st.slider("Maximum visible colony symbols",12,48,36,6)
        if visual!=VISUALS[2]:st.write("The visual maps the finite output range in this analysis window to 0–1. A dot, bar, or thermometer is an educational representation, not a second physical model.")
    xs,ys=graph_samples(expr,analysis,lo,hi);finite=ys[np.isfinite(ys)]
    if not len(finite):st.warning("No finite sample values were found. Try another interval.");return
    bounds=tuple(np.percentile(finite,[2,98]).astype(float))
    st.caption(f"Visual scale uses the 2nd–98th percentile of finite plotted outputs: {num(bounds[0])} to {num(bounds[1])} {output_unit}. Drawing may saturate beyond this scale; numerical values are never clipped.")
    signature=hashlib.sha256((sp.srepr(expr)+str(restrictions)+str((lo,hi,context,source))).encode()).hexdigest()[:16]
    if st.session_state.get("lab_signature")!=signature:
        st.session_state.lab_signature=signature;st.session_state.lab_playing=False;st.session_state.lab_cursor=float(lo)
    # Independent timer reruns only the graph/values/visual, not API fetching or symbolic analysis.
    @st.fragment(run_every=.35)
    def live_view():
        a,b,c,d=st.columns([1,1,1,2])
        if a.button("Pause" if st.session_state.lab_playing else "Play",key="lab_play"):
            st.session_state.lab_playing=not st.session_state.lab_playing
        reset_pressed=b.button("Reset",key="lab_reset")
        if reset_pressed:
            st.session_state.lab_playing=False;st.session_state.lab_cursor=float(lo)
        speed=c.select_slider("Playback speed",options=[.5,1.,2.,4.],value=1.,key="lab_speed")
        if st.session_state.lab_playing:
            step=(hi-lo)/100*speed
            st.session_state.lab_cursor=lo if st.session_state.lab_cursor+step>hi else st.session_state.lab_cursor+step
        slider_key=f"cursor_{signature}"
        # Avoid mutation of an already-rendered widget: synchronize before slider creation.
        if st.session_state.lab_playing or reset_pressed or slider_key not in st.session_state:
            st.session_state[slider_key]=float(st.session_state.lab_cursor)
        point=d.slider("Current input",float(lo),float(hi),key=slider_key,step=max((hi-lo)/1000,1e-7))
        st.session_state.lab_cursor=point
        e,f=st.columns(2)
        start=e.slider("Secant interval start",float(lo),float(hi),float(lo),key=f"secant_a_{signature}",step=max((hi-lo)/1000,1e-7))
        finish=f.slider("Secant interval end",float(lo),float(hi),float(hi),key=f"secant_b_{signature}",step=max((hi-lo)/1000,1e-7))
        permitted=domain_contains(analysis["domain"],point)
        y=evaluate(expr,point) if permitted else np.nan;slope=evaluate(analysis["first"],point) if permitted else np.nan;curve=evaluate(analysis["second"],point) if permitted else np.nan
        ya=evaluate(expr,start) if domain_contains(analysis["domain"],start) else np.nan;yb=evaluate(expr,finish) if domain_contains(analysis["domain"],finish) else np.nan
        avg=(yb-ya)/(finish-start) if abs(finish-start)>1e-12 and np.isfinite([ya,yb]).all() else np.nan
        if not np.isfinite(avg):st.caption("Average rate is undefined when endpoints coincide or an endpoint is outside the real domain.")
        fig=go.Figure()
        if show_function:fig.add_trace(go.Scatter(x=xs,y=ys,name="f(x)",line=dict(color="#147d92",width=3),connectgaps=False))
        if show_first:
            xx,yy=derivative_samples(analysis["first"],analysis,lo,hi);fig.add_trace(go.Scatter(x=xx,y=yy,name="f′(x)",line_color="#e98925",connectgaps=False))
        if show_second:
            xx,yy=derivative_samples(analysis["second"],analysis,lo,hi);fig.add_trace(go.Scatter(x=xx,y=yy,name="f″(x)",line_color="#9333a8",connectgaps=False))
        if np.isfinite(y):fig.add_trace(go.Scatter(x=[point],y=[y],name="Current point",mode="markers",marker=dict(size=13,color="#e34259")))
        if show_secant and np.isfinite(avg):fig.add_trace(go.Scatter(x=[start,finish],y=[ya,yb],name=f"Secant · slope {num(avg)}",mode="lines+markers",line=dict(dash="dash",color="#e98925")))
        if show_tangent and np.isfinite([y,slope]).all():
            tx=np.array([max(lo,point-(hi-lo)*.12),min(hi,point+(hi-lo)*.12)])
            fig.add_trace(go.Scatter(x=tx,y=y+slope*(tx-point),name="Tangent",line_color="#e34259"))
        if level=="Calculus":
            for label,items,color,symbol in [("Critical",analysis["critical"],"#e34259","diamond"),("Inflection",analysis["inflections"],"#9333a8","square")]:
                if items:fig.add_trace(go.Scatter(x=[r["x"] for r in items],y=[r["y"] for r in items],mode="markers",name=label,marker=dict(size=9,color=color,symbol=symbol)))
        for gap in analysis["breaks"]:fig.add_vline(x=gap,line_dash="dot",line_color="#e34259")
        # Robust visible scale keeps near-pole samples from flattening the graph.
        if show_function and np.isfinite(bounds).all():
            span=max(bounds[1]-bounds[0],1.);fig.update_yaxes(range=[bounds[0]-.15*span,bounds[1]+.15*span])
        fig.update_layout(height=480,template="plotly_white",xaxis_title=f"{independent} ({input_unit})",yaxis_title=f"{dependent} ({output_unit}); derivative curves have their own units",legend=dict(orientation="h"))
        left,right=st.columns([2.2,1]);left.plotly_chart(fig,width="stretch",key="lab_graph")
        phase=(point-lo)/(hi-lo);right.plotly_chart(simulation(visual,y,bounds,phase,meta,config),width="stretch",key="lab_visual")
        right.caption("Exact displayed values drive the same current point, tangent, and visual. Plot scale trims extreme pole tails; pan/zoom to inspect them.")
        cards=st.columns(4);cards[0].metric("f(x)",num(y));cards[1].metric("Average rate",num(avg));cards[2].metric("Instantaneous rate",num(slope) if level=="Calculus" else "Calculus mode");cards[3].metric("Second derivative",num(curve) if level=="Calculus" else "Calculus mode")
        if level=="Calculus":st.info(contextual_text(meta,point,y,slope,curve))
        else:
            st.info(f"At {independent} = {num(point)} {input_unit}, {dependent} = {num(y)} {output_unit}. From {num(start)} to {num(finish)}, average change is {num(avg)} {output_unit} per {input_unit}.")
        if show_tangent and np.isfinite([y,slope]).all():st.latex("L(x)="+sp.latex(sp.Float(y)+sp.Float(slope)*(X-sp.Float(point))))
        if np.isfinite(avg):st.caption(f"Secant: ({num(yb)} − {num(ya)}) / ({num(finish)} − {num(start)}) = {num(avg)} {output_unit} per {input_unit}.")
    live_view()
    st.subheader("Intervals and key features")
    trend=pd.DataFrame(analysis["trend"],columns=["Start ≈","End ≈","Behavior"])
    st.dataframe(trend,hide_index=True,width="stretch")
    if level in ("Precalculus","Calculus"):
        st.dataframe(pd.DataFrame(analysis["concavity"],columns=["Start ≈","End ≈","Concavity"]),hide_index=True,width="stretch")
    if level=="Calculus":
        st.write(f"**Root method:** {analysis['root_method']}")
        if analysis["critical"]:st.dataframe(features_frame(analysis["critical"]),hide_index=True,width="stretch")
        else:st.write("No isolated critical points found in this interval.")
        if analysis["inflections"]:st.dataframe(features_frame(analysis["inflections"]),hide_index=True,width="stretch")
        else:st.write("No concavity-changing inflection points found.")
        if analysis["absolute"]:st.dataframe(pd.DataFrame(analysis["absolute"]),hide_index=True,width="stretch")
        st.caption("Local classification uses derivative sign changes, not just zeros. Concavity changes are checked on both sides. Interval tables use open interiors; domain endpoints and excluded points need separate attention.")
    if expr.is_polynomial(X):
        poly=sp.Poly(expr,X);lead=poly.LC();degree=poly.degree()
        if degree>0:st.write(f"**Polynomial end behavior:** degree {degree}; leading coefficient {num(lead)}. As x → +∞, f → {'+∞' if lead>0 else '−∞'}; as x → −∞, f → {'+∞' if ((degree%2==0 and lead>0) or (degree%2==1 and lead<0)) else '−∞'}. This is algebraic behavior, not a prediction about a real population or economy.")
    finite_range=f"Sampled visible outputs range approximately from {num(np.min(finite))} to {num(np.max(finite))} {output_unit}."
    st.caption(finite_range+" This is sampled, not an exact function range; unbounded behavior may occur near excluded inputs.")
    with st.expander("Student questions and mathematical checks"):
        st.markdown("1. Interpret a point and an average rate with correct units.\n2. Compare two intervals: is the rate itself increasing or decreasing?\n3. Explain a local maximum/minimum using a sign change, if one exists.\n4. Verify a proposed inflection point by a change in concavity.\n5. Which inputs are excluded, and why?\n6. What evidence would justify using this model beyond its original interval?")
        st.write("Exact symbolic differentiation applies to the displayed formula. Numerical feature search uses a bounded grid and may miss high-frequency or closely spaced roots; it is not a substitute for proof. Simulations illustrate the formula and are not medical, engineering, or economic validation.")
    # A second set of exports captures the selected function model and key features.
    with st.expander("Export this function model"):
        meta_report={"Context":desc,"Course":level,"Independent quantity":independent,"Input units":input_unit,"Dependent quantity":dependent,"Output units":output_unit,"Formula":sp.sstr(expr),"First derivative":sp.sstr(analysis["first"]),"Second derivative":sp.sstr(analysis["second"]),"Domain in window":sp.sstr(analysis["domain"]),"Root method":analysis["root_method"]}
        sample=pd.DataFrame({"x":xs,"f(x)":ys}).dropna()
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="xlsxwriter",engine_kwargs={"options":{"strings_to_formulas":False,"strings_to_urls":False}}) as writer:
            pd.DataFrame(meta_report.items(),columns=["Field","Value"]).to_excel(writer,sheet_name="Model summary",index=False)
            sample.to_excel(writer,sheet_name="Sampled function",index=False)
            features_frame(analysis["critical"]+analysis["inflections"]).to_excel(writer,sheet_name="Features",index=False)
        fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(xs,ys,color="#147d92");ax.set(xlabel=f"{independent} ({input_unit})",ylabel=f"{dependent} ({output_unit})",title=desc);ax.grid(alpha=.2);fig.tight_layout()
        png=io.BytesIO();fig.savefig(png,format="png",dpi=140);plt.close(fig)
        findings=list(analysis["notes"])+["Feature search: "+analysis["root_method"],"Numerical searches may miss roots; verify conclusions analytically. Visuals illustrate a model rather than validating its real-world assumptions."]
        pdf=export_pdf(meta_report,sample,features_frame(analysis["critical"]+analysis["inflections"]),png.getvalue(),findings,title="Contextual Function Lab Report")
        c=st.columns(3);c[0].download_button("Function samples CSV",sample.to_csv(index=False).encode(),"function_samples.csv","text/csv")
        c[1].download_button("Function analysis Excel",buf.getvalue(),"function_analysis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c[2].download_button("Function analysis PDF",pdf,"function_analysis.pdf","application/pdf")


def main():
    st.set_page_config(page_title="Hispanic Heritage Explorer & Function Lab",page_icon="📈",layout="wide")
    st.markdown("""<style>.block-container{max-width:1500px;padding-top:2rem}h1,h2,h3{color:#173447}div[data-testid=stMetric]{background:#f0f7f8;border-radius:10px;padding:12px} @media print{header,footer,[data-testid=stSidebar],button{display:none!important}.block-container{padding:0!important} }</style>""",unsafe_allow_html=True)
    st.title("Hispanic Heritage Explorer & Function Lab")
    st.caption("Research real country data • Fit and assess models • Explore calculus in changing contexts")
    section=st.sidebar.radio("Workspace",["Historical data","Function Lab","Guide & deployment"],index=1)
    if section=="Historical data":data_explorer()
    elif section=="Function Lab":function_lab()
    else:
        st.header("Getting started")
        st.markdown("**Historical data:** choose country/indicator, inspect source coverage, edit a scenario if desired, select degree, calculate years, compare countries, and export.\n\n**Function Lab:** choose course/context and a suggested, parameter-built, own, or retrieved-data model. Explore the domain, rates, feature tables, and synchronized visual. Use Play/Pause/Reset to animate.\n\n**Heritage inquiry:** connect findings to credible historical/cultural sources. National data are not a measure of cultural worth. Explain what each indicator includes and leaves out.")
        st.subheader("Local / Streamlit deployment")
        st.code("pip install -r requirements.txt\npython -m streamlit run app.py",language="bash")
        st.write("Upload app.py and requirements.txt into the root of a GitHub repository. In Streamlit Community Cloud at share.streamlit.io, create an app, select repository/branch and app.py, then deploy. The filename requirements.txt is required for automatic pip dependency discovery; a file named requirements alone is not recognized.")
        st.subheader("Render")
        st.code("# Build command\npip install -r requirements.txt\n# Start command\npython -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT",language="bash")
        st.write("Use a Python Web Service. Review deployment logs for the first relevant error. Do not commit credentials. The app uses no API key and needs no separate custom module or dataset file.")
        st.markdown("**Sources:** [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries) · [Streamlit dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) · [Render Web Services](https://render.com/docs/web-services)")
        st.write("Verification examples: x^3−3x has extrema at −1 and 1 and an inflection at 0; x^4 has a minimum at 0 but no inflection there; 1/(x−2) excludes 2; log(x) requires x>0; sqrt(x) allows x≥0; x/x retains the original exclusion x≠0.")


if __name__ == "__main__":
    main()
