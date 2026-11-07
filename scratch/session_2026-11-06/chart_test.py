"""Isolate the negative-bar rendering issue: three charts with different number formats."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])
vals = (0.94, -1.22, 0.19, -0.31, -0.61)
variants = [("General, no label fmt", None), ("0.00;-0.00", "0.00;-0.00"), ("+0.00;-0.00;0.00", "+0.00;-0.00;0.00")]
for i, (name, fmt) in enumerate(variants):
    cd = CategoryChartData()
    cd.categories = ["a", "b", "c", "d", "e"]
    cd.add_series(name, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.3 + i * 4.3), Inches(0.5), Inches(4.1), Inches(5.5), cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = name
    plot = ch.plots[0]
    plot.has_data_labels = True
    if fmt:
        plot.data_labels.number_format = fmt
        plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
prs.save(r"C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/render/chart_test.pptx")
print("ok")
