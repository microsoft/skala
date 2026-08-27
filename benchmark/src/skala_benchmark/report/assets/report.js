(() => {
  "use strict";

  const report = JSON.parse(document.getElementById("report-data").textContent);
  const fits = JSON.parse(document.getElementById("report-fits").textContent);
  const d3 = window.d3;
  const tooltip = document.getElementById("chart-tooltip");
  const functionLabels = Object.fromEntries(
    report.meta.functionals.map((functional) => [functional, functional]),
  );
  const fallbackColors = ["#49c5b1", "#8661c5", "#ffb900", "#0078d4", "#ffa38b", "#b1b3b3"];
  const colors = report.meta.functionals.map((_, index) => {
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue(`--series-${index + 1}`)
      .trim();
    return value || fallbackColors[index % fallbackColors.length];
  });
  const state = {
    environment: report.meta.initial_selection.environment,
    basis: report.meta.initial_selection.basis,
    xMode: "num_aos",
  };
  const missingFitWarnings = new Set();
  let tooltipPinned = false;

  function renderMath() {
    if (typeof window.katex === "undefined") return;
    document.querySelectorAll(".tex").forEach((element) => {
      const tex = element.getAttribute("data-tex") ?? element.textContent;
      try {
        window.katex.render(tex, element, {
          throwOnError: false,
          displayMode: element.classList.contains("tex--display"),
          output: "mathml",
        });
      } catch (error) {
        /* leave the raw TeX in place if rendering fails */
      }
    });
  }

  const environmentItems = report.meta.environments.map((environment) => ({
    key: environment.env_id,
    label: environment.control_label || environment.label,
  }));
  const basisItems = report.meta.bases.map((basis) => ({ key: basis, label: basis }));
  const xItems = [
    { key: "num_aos", label: report.meta.x_modes.num_aos.label },
    { key: "grid_size", label: report.meta.x_modes.grid_size.label },
  ];

  const controlGroups = [];

  function buildSegmentGroup(elements, items, getValue, setValue) {
    const group = { elements, getValue };
    elements.forEach((element) => {
      items.forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = item.label;
        button.dataset.key = item.key;
        button.setAttribute("role", "radio");
        button.addEventListener("click", () => {
          setValue(item.key);
          syncControls();
          renderAll();
        });
        button.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            setValue(event.currentTarget.dataset.key);
            syncControls();
            renderAll();
          }
        });
        element.appendChild(button);
      });
    });
    controlGroups.push(group);
    updateSegmentGroup(group);
  }

  function updateSegmentGroup(group) {
    const selected = group.getValue();
    group.elements.forEach((element) => updateSegment(element, selected));
  }

  function updateSegment(element, selected) {
    const buttons = [...element.querySelectorAll("button")];
    buttons.forEach((button) => {
      const active = button.dataset.key === selected;
      button.setAttribute("aria-checked", active ? "true" : "false");
    });
  }

  function syncControls() {
    controlGroups.forEach(updateSegmentGroup);
  }

  function cycle(items, current) {
    const index = Math.max(
      0,
      items.findIndex((item) => item.key === current),
    );
    return items[(index + 1) % items.length]?.key;
  }

  const compositionSeries = report.meta.composition_buckets || {};
  const compositionColors = (buckets) =>
    buckets.map((_, index) => {
      const value = getComputedStyle(document.documentElement)
        .getPropertyValue(`--series-${index + 1}`)
        .trim();
      return value || fallbackColors[index % fallbackColors.length];
    });

  // One panel per functional, in a container that the template provides.
  function buildFunctionalPanels(containerId, chartClass, dataset) {
    const container = document.getElementById(containerId);
    if (!container) {
      return;
    }
    report.meta.functionals.forEach((functional) => {
      const block = document.createElement("article");
      block.className = "composition-block";
      const heading = document.createElement("h3");
      heading.textContent = functionLabels[functional] || functional;
      const chart = document.createElement("div");
      chart.className = `chart ${chartClass}`;
      chart.dataset.functional = functional;
      Object.assign(chart.dataset, dataset || {});
      block.append(heading, chart);
      container.appendChild(block);
    });
  }

  function renderAll() {
    document.querySelectorAll(".chart[data-metric]").forEach((element) => {
      drawChart(element, element.dataset.metric);
    });
    document.querySelectorAll(".composition-chart").forEach((element) => {
      drawComposition(element, element.dataset.functional, element.dataset.series);
    });
  }

  function drawChart(element, metric) {
    // Metrics that were fitted against grid size get an x-axis toggle.
    const hasXToggle = Boolean(report.domains[metric]?.grid_size);
    const xMode = hasXToggle ? state.xMode : "num_aos";
    const domain = report.domains[metric][xMode];
    const width = Math.max(300, Math.round(element.clientWidth || 500));
    const height = width < 420 ? 310 : 340;
    const margin = { top: 12, right: 82, bottom: hasXToggle ? 26 : 44, left: 52 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const rows = report.points.filter(
      (row) =>
        row.env_id === state.environment &&
        row.basis === state.basis &&
        row.metric === metric &&
        row.x_axis === xMode &&
        Number(row.x) > 0 &&
        Number(row.y) > 0,
    );

    element.replaceChildren();
    const svg = d3
      .select(element)
      .append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr(
        "aria-label",
        `${report.meta.metrics[metric].label} versus ${report.meta.x_modes[xMode].label}`,
      );
    const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const x = d3.scaleLog().domain(domain.x).range([0, innerWidth]);
    const y = d3.scaleLog().domain(domain.y).range([innerHeight, 0]);

    plot
      .append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(-innerHeight).tickFormat(""));
    plot
      .append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(""));
    plot
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5, "~s").tickSizeOuter(0));
    plot
      .append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(y).ticks(5, "~s").tickSizeOuter(0));
    if (!hasXToggle) {
      svg
        .append("text")
        .attr("class", "axis-label")
        .attr("x", margin.left + innerWidth / 2)
        .attr("y", height - 8)
        .attr("text-anchor", "middle")
        .text(report.meta.x_modes[xMode].label);
    }
    svg
      .append("text")
      .attr("class", "axis-label")
      .attr("transform", "rotate(-90)")
      .attr("x", -(margin.top + innerHeight / 2))
      .attr("y", 14)
      .attr("text-anchor", "middle")
      .text(report.meta.metrics[metric].unit);

    if (!rows.length) {
      plot
        .append("text")
        .attr("class", "empty-message")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight / 2)
        .text("No measurements for this selection");
      return;
    }

    const seriesLayer = plot.append("g").attr("class", "series-layer");
    const labelLayer = plot.append("g").attr("class", "label-layer");
    const seriesLabels = [];
    const seriesGroups = new Map();

    report.meta.functionals.forEach((functional, index) => {
      const series = rows
        .filter((row) => row.functional === functional)
        .sort((left, right) => left.x - right.x);
      if (!series.length) {
        return;
      }
      const seriesGroup = seriesLayer
        .append("g")
        .attr("class", "functional-series")
        .attr("data-functional", functional);
      const fitLayer = seriesGroup.append("g").attr("class", "fit-layer");
      const pointLayer = seriesGroup.append("g").attr("class", "point-layer");
      seriesGroups.set(functional, seriesGroup);
      const color = colors[index];
      const segments = fits
        .filter(
          (fit) =>
            fit.env_id === state.environment &&
            fit.basis === state.basis &&
            fit.functional === functional &&
            fit.metric === metric &&
            fit.x_axis === xMode,
        )
        .slice()
        .sort((left, right) => Number(left.x_start) - Number(right.x_start));
      const fitLine = d3
        .line()
        .x((point) => x(point.x))
        .y((point) => y(point.y));
      const yAt = (segment, value) =>
        10 ** (Number(segment.intercept) + Number(segment.slope) * Math.log10(value));
      const segEndpoints = (segment) => {
        const xs = Number(segment.x_start);
        const xe = Number(segment.x_end);
        return { start: [x(xs), y(yAt(segment, xs))], end: [x(xe), y(yAt(segment, xe))] };
      };
      const darkerColorObj = d3.color(color);
      const darkerColor = darkerColorObj ? darkerColorObj.darker(0.9).formatHex() : color;
      segments.forEach((segment) => {
        const fitPoints = [Number(segment.x_start), Number(segment.x_end)].map((value) => ({
          x: value,
          y: yAt(segment, value),
        }));
        const linePath = fitLayer
          .append("path")
          .datum(fitPoints)
          .attr("class", "fit-line")
          .attr("stroke", color)
          .attr("d", fitLine);
        const activate = () => linePath.classed("fit-line--active", true).attr("stroke", darkerColor);
        const deactivate = () => linePath.classed("fit-line--active", false).attr("stroke", color);
        fitLayer
          .append("path")
          .datum(fitPoints)
          .attr("class", "fit-hit")
          .attr("d", fitLine)
          .on("pointerenter", (event) => {
            activate();
            showSlopeTooltip(event, segment, functional);
          })
          .on("pointermove", (event) => positionTooltip(event.clientX, event.clientY))
          .on("pointerleave", () => {
            deactivate();
            if (!tooltipPinned) {
              hideTooltip();
            }
          })
          .on("pointerdown", (event) => {
            event.stopPropagation();
            tooltipPinned = true;
            activate();
            showSlopeTooltip(event, segment, functional);
          });
      });
      const kinkColor = color;
      const emphasizeKink = (anchor, toward) => {
        const dx = toward[0] - anchor[0];
        const dy = toward[1] - anchor[1];
        const length = Math.hypot(dx, dy);
        if (length < 1e-6) {
          return;
        }
        const reach = Math.min(16, length * 0.45);
        const tipX = anchor[0] + (dx / length) * reach;
        const tipY = anchor[1] + (dy / length) * reach;
        fitLayer
          .append("path")
          .attr("class", "fit-kink")
          .attr("stroke", kinkColor)
          .attr("d", `M${anchor[0]},${anchor[1]}L${tipX},${tipY}`);
      };
      for (let index = 0; index + 1 < segments.length; index += 1) {
        const left = segEndpoints(segments[index]);
        const right = segEndpoints(segments[index + 1]);
        emphasizeKink(left.end, left.start);
        emphasizeKink(right.start, right.end);
      }
      if (!segments.length) {
        warnMissingFit(metric, xMode, functional);
      }

      pointLayer
        .selectAll(`circle.point-${index}`)
        .data(series)
        .join("circle")
        .attr("class", "point")
        .attr("cx", (row) => x(row.x))
        .attr("cy", (row) => y(row.y))
        .attr("r", 4.2)
        .attr("fill", color)
        .attr("tabindex", 0)
        .attr(
          "aria-label",
          (row) =>
            `${row.molecule}, ${functionLabels[row.functional] || row.functional}, ` +
            `${formatValue(row.y)} ${report.meta.metrics[metric].unit}`,
        )
        .on("pointerenter focus", (event, row) => showTooltip(event, row, metric))
        .on("pointermove", (event) => positionTooltip(event.clientX, event.clientY))
        .on("pointerleave blur", () => {
          if (!tooltipPinned) {
            hideTooltip();
          }
        })
        .on("pointerdown", (event, row) => {
          event.stopPropagation();
          tooltipPinned = true;
          showTooltip(event, row, metric);
        });

      // Anchor the label on the fit line's right end (fall back to the last
      // scatter point only when no fit is available) so it hugs the curve.
      let anchorX;
      let anchorY;
      if (segments.length) {
        const [endX, endY] = segEndpoints(segments[segments.length - 1]).end;
        anchorX = endX;
        anchorY = endY;
      } else {
        const endpoint = series.reduce((best, row) => (row.x > best.x ? row : best));
        anchorX = x(endpoint.x);
        anchorY = y(endpoint.y);
      }
      seriesLabels.push({
        functional,
        text: functionLabels[functional] || functional,
        color,
        anchorX,
        anchorY,
        y: anchorY,
      });
    });

    const restoreSeriesOrder = () => {
      report.meta.functionals.forEach((functional) => seriesGroups.get(functional)?.raise());
    };
    placeSeriesLabels(seriesLabels, innerHeight, 13);
    seriesLabels.forEach((label) => {
      const focusSeries = () => seriesGroups.get(label.functional)?.raise();
      labelLayer
        .append("text")
        .attr("class", "series-label")
        .attr("tabindex", 0)
        .attr("data-functional", label.functional)
        .attr("aria-label", `Bring ${label.text} measurements to the front`)
        .attr("x", Math.min(label.anchorX + 5, innerWidth + 6))
        .attr("y", label.y)
        .attr("fill", label.color)
        .attr("dominant-baseline", "middle")
        .text(label.text)
        .on("pointerenter focus", focusSeries)
        .on("pointerleave blur", restoreSeriesOrder);
    });
  }

  function drawComposition(element, functional, series) {
    const buckets = compositionSeries[series] || [];
    const bucketColors = compositionColors(buckets);
    const domain = report.composition_domain[series];
    const width = Math.max(300, Math.round(element.clientWidth || 500));
    const height = width < 420 ? 230 : 250;
    const margin = { top: 12, right: width < 460 ? 104 : 132, bottom: 46, left: 54 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const records = (report.composition[series] || [])
      .filter(
        (row) =>
          row.env_id === state.environment &&
          row.basis === state.basis &&
          row.functional === functional &&
          Number(row.x) > 0,
      )
      .sort((left, right) => left.x - right.x);

    element.replaceChildren();
    const svg = d3
      .select(element)
      .append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr(
        "aria-label",
        `Time composition for ${functionLabels[functional] || functional}`,
      );
    const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const sharedRecords = (report.composition[series] || []).filter(
      (row) =>
        row.env_id === state.environment &&
        row.basis === state.basis &&
        Number(row.x) > 0,
    );
    const xExtent = d3.extent(sharedRecords, (row) => Number(row.x));
    const xDomain =
      xExtent[0] && xExtent[1] && xExtent[0] !== xExtent[1] ? xExtent : domain.x;
    const x = d3.scaleLog().domain(xDomain).range([0, innerWidth]);
    const y = d3.scaleLinear().domain(domain.y).range([innerHeight, 0]);

    plot
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5, "~s").tickSizeOuter(0));
    plot
      .append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")).tickSizeOuter(0));
    svg
      .append("text")
      .attr("class", "axis-label")
      .attr("x", margin.left + innerWidth / 2)
      .attr("y", height - 8)
      .attr("text-anchor", "middle")
      .text(report.meta.x_modes.num_aos.label);

    if (!records.length) {
      plot
        .append("text")
        .attr("class", "empty-message")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight / 2)
        .text("No measurements for this selection");
      return;
    }

    const smooth = (report.composition_smooth[series] || []).find(
      (row) =>
        row.env_id === state.environment &&
        row.basis === state.basis &&
        row.functional === functional,
    );
    const useSmooth =
      smooth && Array.isArray(smooth.x) && smooth.x.length >= 2;

    buckets.forEach((bucket, index) => {
      const band = useSmooth
        ? smooth.x.map((xv, k) => ({
            x: xv,
            lower: smooth.cumulative[index][k],
            upper: smooth.cumulative[index + 1][k],
          }))
        : records.map((row) => {
            const values = row.y;
            const lower = values
              .slice(0, index)
              .reduce((sum, value) => sum + value, 0);
            return { x: row.x, lower, upper: lower + values[index] };
          });
      plot
        .append("path")
        .datum(band)
        .attr("class", "composition-area")
        .attr("fill", bucketColors[index])
        .attr(
          "d",
          d3
            .area()
            .curve(useSmooth ? d3.curveLinear : d3.curveMonotoneX)
            .x((point) => x(point.x))
            .y0((point) => y(point.lower))
            .y1((point) => y(point.upper)),
        );
    });

    const cumulative = [];
    if (useSmooth) {
      const edge = smooth.x.length - 1;
      buckets.forEach((bucket, index) => {
        cumulative.push(smooth.cumulative[index + 1][edge]);
      });
    } else {
      const last = records[records.length - 1];
      last.y.reduce((sum, value) => {
        cumulative.push(sum + value);
        return sum + value;
      }, 0);
    }
    const labels = buckets.map((bucket, index) => {
      const lower = index === 0 ? 0 : cumulative[index - 1];
      const upper = cumulative[index];
      return {
        text: bucket.short || bucket.label,
        color: bucketColors[index],
        y: (y(lower) + y(upper)) / 2,
        share: upper - lower,
      };
    });
    declutterLabels(labels, innerHeight, 12);
    plot
      .append("g")
      .attr("class", "composition-labels")
      .selectAll("text")
      .data(labels.filter((label) => label.share >= 0.02))
      .join("text")
      .attr("x", innerWidth + 6)
      .attr("y", (label) => label.y)
      .attr("fill", (label) => label.color)
      .attr("dominant-baseline", "middle")
      .text((label) => label.text);

    const bisect = d3.bisector((row) => row.x).center;
    plot
      .append("rect")
      .attr("class", "composition-overlay")
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("fill", "none")
      .style("pointer-events", "all")
      .on("pointermove", (event) => {
        const [pointerX] = d3.pointer(event);
        const value = x.invert(pointerX);
        const row = records[Math.min(records.length - 1, Math.max(0, bisect(records, value)))];
        showCompositionTooltip(event, functional, row, series);
      })
      .on("pointerleave", () => {
        if (!tooltipPinned) {
          hideTooltip();
        }
      });
  }

  function placeSeriesLabels(labels, height, minGap) {
    const top = 6;
    const bottom = height - 2;
    if (labels.length < 2) {
      labels.forEach((label) => {
        label.y = Math.max(top, Math.min(bottom, label.anchorY));
      });
      return;
    }
    // Sort by preferred (line-endpoint) position, then push labels apart just
    // enough to clear the minimum gap, keeping them near their own line.
    labels.sort((left, right) => left.anchorY - right.anchorY);
    labels.forEach((label) => {
      label.y = label.anchorY;
    });
    for (let index = 1; index < labels.length; index += 1) {
      if (labels[index].y - labels[index - 1].y < minGap) {
        labels[index].y = labels[index - 1].y + minGap;
      }
    }
    // If the stack overflowed the bottom, pin the last label and resolve upward.
    if (labels[labels.length - 1].y > bottom) {
      labels[labels.length - 1].y = bottom;
      for (let index = labels.length - 2; index >= 0; index -= 1) {
        if (labels[index + 1].y - labels[index].y < minGap) {
          labels[index].y = labels[index + 1].y - minGap;
        }
      }
    }
    // Final top-edge clamp, pushing subsequent labels back down if needed.
    for (let index = 0; index < labels.length; index += 1) {
      if (labels[index].y < top) {
        labels[index].y = top;
      }
      if (
        index + 1 < labels.length &&
        labels[index + 1].y - labels[index].y < minGap
      ) {
        labels[index + 1].y = labels[index].y + minGap;
      }
    }
  }

  function declutterLabels(labels, height, minGap) {
    const visible = labels.filter((label) => label.share >= 0.02);
    visible.sort((left, right) => left.y - right.y);
    for (let index = 1; index < visible.length; index += 1) {
      if (visible[index].y - visible[index - 1].y < minGap) {
        visible[index].y = visible[index - 1].y + minGap;
      }
    }
    for (let index = visible.length - 1; index > 0; index -= 1) {
      if (visible[index].y > height) {
        visible[index].y = height;
      }
      if (visible[index].y - visible[index - 1].y < minGap) {
        visible[index - 1].y = visible[index].y - minGap;
      }
    }
    visible.forEach((label) => {
      label.y = Math.max(6, Math.min(height, label.y));
    });
  }

  function showCompositionTooltip(event, functional, row, series) {
    tooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = `${functionLabels[functional] || functional} · ${formatInteger(row.x)} AOs`;
    const list = document.createElement("dl");
    (compositionSeries[series] || []).forEach((bucket, index) => {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = bucket.label;
      detail.textContent = formatPercent(row.y[index]);
      list.append(term, detail);
    });
    tooltip.append(heading, list);
    tooltip.hidden = false;
    positionTooltip(event.clientX, event.clientY);
  }

  function warnMissingFit(metric, xMode, functional) {    const key = `${state.environment}/${state.basis}/${functional}/${metric}/${xMode}`;
    if (!missingFitWarnings.has(key)) {
      missingFitWarnings.add(key);
      console.warn(`No precomputed fit for ${key}; fitted line omitted.`);
    }
  }

  function showTooltip(event, row, metric) {
    const fields = [
      ["#AOs", formatInteger(row.num_aos)],
      ["Grid points", formatInteger(row.grid_size)],
      ["Ansatz", row.ansatz || "—"],
      ["Electrons", formatInteger(row.electrons)],
      ["Atoms", formatInteger(row.atoms)],
      ["Basis", row.basis],
      ["Functional", functionLabels[row.functional] || row.functional],
      ["SCF iterations", formatInteger(row.scf_iterations)],
      [
        report.meta.metrics[metric].label,
        `${formatValue(row.y)} ${report.meta.metrics[metric].unit}`,
      ],
      ["Total energy", row.energy == null ? "—" : `${Number(row.energy).toFixed(8)} Eh`],
    ];
    if (row.warmup_ratio != null) {
      fields.push([
        "First-iteration warmup",
        `${formatValue(row.warmup_ratio)}x steady state`,
      ]);
    }
    tooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = row.molecule;
    const list = document.createElement("dl");
    fields.forEach(([label, value]) => {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      list.append(term, detail);
    });
    tooltip.append(heading, list);
    tooltip.hidden = false;
    positionTooltip(event.clientX, event.clientY);
  }

  function showSlopeTooltip(event, segment, functional) {
    const axisSuffix = state.xMode === "grid_size" ? "grid pts" : "AOs";
    const fields = [
      ["Functional", functionLabels[functional] || functional],
      ["Log–log slope", Number(segment.slope).toFixed(2)],
      [
        "Range",
        `${formatInteger(segment.x_start)} – ${formatInteger(segment.x_end)} ${axisSuffix}`,
      ],
      ["Fit", segment.continuous ? "continuous" : "discontinuous"],
    ];
    tooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = "Fitted segment";
    const list = document.createElement("dl");
    fields.forEach(([label, value]) => {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      list.append(term, detail);
    });
    tooltip.append(heading, list);
    tooltip.hidden = false;
    positionTooltip(event.clientX, event.clientY);
  }

  function positionTooltip(clientX, clientY) {
    if (tooltip.hidden) {
      return;
    }
    const padding = 12;
    const rect = tooltip.getBoundingClientRect();
    let left = clientX + 14;
    let top = clientY + 14;
    if (left + rect.width > window.innerWidth - padding) {
      left = clientX - rect.width - 14;
    }
    if (top + rect.height > window.innerHeight - padding) {
      top = clientY - rect.height - 14;
    }
    tooltip.style.left = `${Math.max(padding, left)}px`;
    tooltip.style.top = `${Math.max(padding, top)}px`;
  }

  function hideTooltip() {
    tooltip.hidden = true;
    tooltipPinned = false;
  }

  function formatInteger(value) {
    return value == null ? "—" : Math.round(Number(value)).toLocaleString();
  }

  function formatValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    return number >= 100 ? number.toFixed(0) : number >= 10 ? number.toFixed(1) : number.toFixed(2);
  }

  function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
  }

  buildSegmentGroup(
    [document.getElementById("environment-control")].filter(Boolean),
    environmentItems,
    () => state.environment,
    (value) => {
      state.environment = value;
    },
  );
  buildSegmentGroup(
    [document.getElementById("basis-control")].filter(Boolean),
    basisItems,
    () => state.basis,
    (value) => {
      state.basis = value;
    },
  );
  buildSegmentGroup(
    [...document.querySelectorAll(".x-axis-control")],
    xItems,
    () => state.xMode,
    (value) => {
      state.xMode = value;
    },
  );
  buildFunctionalPanels("composition-charts", "composition-chart", {
    series: "cycle",
  });
  buildFunctionalPanels("run-composition-charts", "composition-chart", {
    series: "run",
  });
  renderAll();
  renderMath();

  const controlBar = document.querySelector(".control-bar");
  const sentinel = document.querySelector(".control-bar__sentinel");
  if (controlBar && sentinel) {
    const updateStuck = () => {
      const stuck = sentinel.getBoundingClientRect().top <= 0;
      controlBar.classList.toggle("is-stuck", stuck);
    };
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        updateStuck();
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    updateStuck();
  }

  document.addEventListener("pointerdown", hideTooltip);
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
      return;
    }
    const key = event.key.toLowerCase();
    if (key === "e") {
      state.environment = cycle(environmentItems, state.environment);
    } else if (key === "b") {
      state.basis = cycle(basisItems, state.basis);
    } else if (key === "x") {
      state.xMode = cycle(xItems, state.xMode);
    } else if (event.key === "Escape") {
      hideTooltip();
      return;
    } else {
      return;
    }
    event.preventDefault();
    syncControls();
    renderAll();
  });

  let resizeTimer;
  const observer = new ResizeObserver(() => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(renderAll, 100);
  });
  document.querySelectorAll(".chart").forEach((chart) => observer.observe(chart));
})();
