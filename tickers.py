"""Ticker universe for the portfolio pickers.

The dashboard is GBP-denominated, so the curated list leans on London-listed
names, then adds the indices, gilts and commodity proxies that are useful as
hedge or comparison instruments.

The list is a convenience, not a constraint: every picker built on it passes
``accept_new_options=True``, so any symbol Yahoo Finance recognises can be
typed straight in.
"""

import streamlit as st

TICKER_UNIVERSE = [
    # FTSE 100: banks and financials
    "HSBA.L", "LLOY.L", "BARC.L", "NWG.L", "STAN.L",
    "PRU.L", "LGEN.L", "AV.L", "III.L",
    # FTSE 100: energy and mining
    "BP.L", "SHEL.L", "RIO.L", "AAL.L", "GLEN.L", "ANTO.L",
    # FTSE 100: healthcare and consumer
    "AZN.L", "GSK.L", "ULVR.L", "DGE.L", "BATS.L",
    "IMB.L", "RKT.L", "TSCO.L", "SBRY.L", "CPG.L",
    # FTSE 100: industrials, telecoms, utilities
    "VOD.L", "BT-A.L", "NG.L", "SSE.L", "REL.L",
    "RR.L", "BA.L", "EXPN.L",
    # Indices
    "^FTSE", "^FTMC", "^GSPC", "^IXIC", "^DJI", "^VIX",
    # ETFs, gilts and commodity proxies
    "ISF.L", "VUSA.L", "VWRL.L", "IGLT.L", "GLD", "TLT",
    # US mega caps
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
]


def normalise(ticker):
    """Yahoo Finance symbols are upper case; typed input may not be."""
    return (ticker or "").strip().upper()


def options(*include):
    """Sorted universe, unioned with any extra symbols that must stay selectable.

    A symbol the user typed is not in the curated universe, so it has to be
    folded back into the options on the next rerun or Streamlit would drop the
    widget's own value.
    """
    extra = {normalise(t) for t in include if normalise(t)}
    return sorted(set(TICKER_UNIVERSE) | extra)


def picker(label, key, extra_keys=()):
    """Searchable ticker picker that also accepts symbols outside the list.

    Starts empty. Typing filters the list; typing something not in the list
    adds it as an option. Returns "" while nothing is selected.
    """
    # A symbol typed into the box is stored verbatim, so "rmv.l" would show
    # lower case in the sidebar while every table and chart shows "RMV.L".
    # Rewriting the stored value before the widget renders keeps them in step.
    stored = st.session_state.get(key)
    if stored and normalise(stored) != stored:
        st.session_state[key] = normalise(stored)

    session_values = [st.session_state.get(k) for k in (key, *extra_keys)]
    selected = st.selectbox(
        label,
        options=options(*session_values),
        index=None,
        format_func=normalise,
        accept_new_options=True,
        placeholder="Search or type a ticker",
        key=key,
    )
    return normalise(selected)
