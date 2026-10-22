"""Manuscript Figure 2 (PropGate from a user's perspective). Output: fig2_user_workflow.pdf/.png in the current folder."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'Liberation Sans'
BLUE, ORANGE, GREEN, PURPLE, PINK, GREY = '#2f6fd6', '#d9822b', '#2e8b57', '#6b5bd2', '#c2417a', '#4a5058'
FILL = {'blue': '#eef4ff', 'orange': '#fff4e8', 'green': '#ecf7ef', 'purple': '#f2efff', 'pink': '#fdeef4', 'grey': '#f4f5f7'}

fig = plt.figure(figsize=(13, 6.1)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 61); ax.axis('off')


def box(x, y, w, h, edge, fill, title, lines, tsize=12.5, lsize=10.6, num=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fill, ec=edge, lw=1.4))
    ty = y + h - 2.6
    if num is not None:
        ax.add_patch(plt.Circle((x + 2.4, ty + 0.1), 1.55, color=edge)); ax.text(x + 2.4, ty + 0.05, str(num), ha='center', va='center', color='white', fontsize=10, weight='bold')
        ax.text(x + 4.6, ty, title, ha='left', va='center', color=edge, fontsize=tsize, weight='bold')
    else:
        ax.text(x + w / 2, ty, title, ha='center', va='center', color=edge, fontsize=tsize, weight='bold')
    for i, s in enumerate(lines):
        ax.text(x + 1.2, ty - 3.4 - 2.8 * i, s, ha='left', va='center', color='#111', fontsize=lsize)


def arrow(x0, y0, x1, y1, c=GREY, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.5, linestyle=ls, shrinkA=0, shrinkB=0))


ax.text(65, 58.6, 'PropGate: applying the method to a new medicinal plant collection', fontsize=15, weight='bold', color='#1f2328', ha='center', va='center')

# panel background
ax.add_patch(FancyBboxPatch((1, 15.5), 128, 40.5, boxstyle='round,pad=0,rounding_size=1.5', fc=FILL['grey'], ec='none'))
y, h = 19, 33.5
box(2.5, y, 22, h, BLUE, FILL['blue'], 'Inputs', [
    'Reference set: b labeled', 'images per known species', '(evaluated with b = 4 and 16)', '',
    'Unlabeled archive: any size;', 'may contain unknown species;', 'labels not needed', '',
    'Query photos to identify'], num=1)
box(27.5, y, 21.5, h, PURPLE, FILL['purple'], 'Embed once', [
    'Frozen BioCLIP 2 image', 'encoder; no training', '', 'Resize 224, center crop,', 'L2-normalized 768-d features', '',
    'GPU recommended,', 'CPU possible'], num=2)
box(52, y, 23.5, h, ORANGE, FILL['orange'], 'Calibrate gate', [
    'Logistic regression on the', 'reference set only', '',
    '4-fold out-of-fold MSP on', 'references; threshold τ =', '10th percentile', '',
    'No unknown examples and', 'no extra labels required'], num=3)
box(78.5, y, 23.5, h, BLUE, FILL['blue'], 'Propagate labels', [
    'k-NN cosine graph over', 'references + archive (k = 10)', '',
    'Iterative label propagation,', 'α = 0.9 (b = 4), 0.8 (b = 16),', 'references clamped', '',
    'CPU only; queries stay', 'outside the graph'], num=4)
box(105, y, 22.5, h, GREEN, FILL['green'], 'Identify or refer', [
    'Query class: weighted vote', 'of its 10 archive neighbours', '',
    'Gate score: reference-head', 'MSP s(q)', ''], num=5)
for x0, x1 in [(24.9, 27.2), (49.4, 51.7), (75.9, 78.2), (102.4, 104.7)]:
    arrow(x0, y + h / 2, x1, y + h / 2)
# outcomes inside box 5
ax.add_patch(FancyBboxPatch((106.3, 26.2), 9.4, 4.4, boxstyle='round,pad=0.15,rounding_size=0.8', fc='white', ec=GREEN, lw=1.2))
ax.text(111, 29.2, 's(q) ≥ τ', ha='center', va='center', fontsize=9.8, color=GREEN, weight='bold'); ax.text(111, 27.3, 'report species', ha='center', va='center', fontsize=9.8, color='#111')
ax.add_patch(FancyBboxPatch((116.9, 26.2), 9.6, 4.4, boxstyle='round,pad=0.15,rounding_size=0.8', fc='white', ec=PINK, lw=1.2))
ax.text(121.7, 29.2, 's(q) < τ', ha='center', va='center', fontsize=9.8, color=PINK, weight='bold'); ax.text(121.7, 27.3, 'refer to expert', ha='center', va='center', fontsize=9.8, color='#111')
ax.text(116.3, 22.3, 'Gate and classifier are decoupled:', ha='center', va='center', fontsize=9.8, color=GREY)
ax.text(116.3, 20.6, 'τ can be moved to a target risk', ha='center', va='center', fontsize=9.8, color=GREY)

# bottom strip: defaults and evaluation protocol
ax.add_patch(FancyBboxPatch((1, 1.2), 63, 12.4, boxstyle='round,pad=0,rounding_size=1.5', fc='white', ec=BLUE, lw=1.2))
ax.text(3, 11.3, 'Defaults (fixed on 60 development classes; no tuning on new data)', fontsize=10.5, weight='bold', color=BLUE, va='center')
for i, s in enumerate(['k = 10 neighbours;  α = 0.9 at 4 labels/class, 0.8 at 16;  ≤ 100 iterations',
                       'Reference head: multinomial LR, C = 1000;  gate quantile = 10%',
                       'Sensitivity of k and α and of the gate quantile: Results 3.7 and Supplementary Section S7']):
    ax.text(3, 8.2 - 2.6 * i, '• ' + s, fontsize=10.6, va='center', color='#111')
ax.add_patch(FancyBboxPatch((66, 1.2), 63, 12.4, boxstyle='round,pad=0,rounding_size=1.5', fc='white', ec=ORANGE, lw=1.2))
ax.text(68, 11.3, 'Evaluation protocol (to compare alternatives on your data)', fontsize=10.5, weight='bold', color=ORANGE, va='center')
for i, s in enumerate(['Conceal pool/query labels; split known / pool-unknown / query-only species',
                       'Fixed-known gradient (0–50% unknowns) + fixed-total control',
                       'Report known accuracy and gate coverage / unknown acceptance / risk separately']):
    ax.text(68, 8.2 - 2.6 * i, '• ' + s, fontsize=10.6, va='center', color='#111')
fig.savefig('fig2_user_workflow.pdf'); fig.savefig('fig2_user_workflow.png', dpi=200)
print('ok')
