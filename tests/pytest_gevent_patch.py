"""
Pytest plugin to patch gevent early.
"""

import gevent.monkey

# Patch immediately on import
gevent.monkey.patch_all()
