import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import * as d3fetch from 'd3-fetch';

const D3Visualization = ({ filepath }) => {
  const svgRef = useRef();
  const [data, setData] = useState(null);
  filepath= './image_matrix.json';

  // Load the data from the given filepath
  useEffect(() => {
    if (filepath) {
      d3fetch.json(filepath)
        .then(loadedData => {
          setData(loadedData);
        })
        .catch(error => {
          console.error('Error loading the JSON file:', error);
        });
    }
  }, [filepath]);

  // Visualize the data once it's loaded
  useEffect(() => {
    if (data) {
      // Dimensions and margins
      const margin = { top: 50, right: 50, bottom: 50, left: 50 };
      const width = 800 - margin.left - margin.right;
      const height = 800 - margin.top - margin.bottom;

      // Clear any previous SVG content
      d3.select(svgRef.current).selectAll('*').remove();

      // Create SVG container
      const svg = d3.select(svgRef.current)
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // Define the color scale
      const colorScale = d3.scaleSequential(d3.interpolateViridis)
        .domain([0, 1]);

      // Calculate the number of points and grid size
      const numPoints = d3.max(data, d => Math.max(d.p, d.q)) + 1;
      const gridSize = Math.floor(width / numPoints);

      // Create heatmap
      svg.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => d.p * gridSize)
        .attr('y', d => d.q * gridSize)
        .attr('width', gridSize)
        .attr('height', gridSize)
        .style('fill', d => colorScale(d.value));

      // Add X axis labels
      const xLabels = Array.from(Array(numPoints).keys());
      svg.append('g')
        .selectAll('text')
        .data(xLabels)
        .enter()
        .append('text')
        .attr('x', (d, i) => i * gridSize + gridSize / 2)
        .attr('y', height + 20)
        .attr('text-anchor', 'middle')
        .text(d => d);

      // Add Y axis labels
      svg.append('g')
        .selectAll('text')
        .data(xLabels)
        .enter()
        .append('text')
        .attr('x', -20)
        .attr('y', (d, i) => i * gridSize + gridSize / 2)
        .attr('text-anchor', 'middle')
        .text(d => d);

      // Add a color legend
      const legendWidth = 20;
      const legendHeight = 200;
      const legendData = Array.from({ length: 100 }, (_, i) => i / 100);

      const legend = svg.append('g')
        .attr('transform', `translate(${width + 40},${margin.top})`);

      legend.selectAll('rect')
        .data(legendData)
        .enter()
        .append('rect')
        .attr('x', 0)
        .attr('y', (d, i) => i * legendHeight / legendData.length)
        .attr('width', legendWidth)
        .attr('height', legendHeight / legendData.length)
        .style('fill', d => colorScale(d));

      legend.append('text')
        .attr('x', 0)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .text('Value');

      legend.append('text')
        .attr('x', legendWidth + 5)
        .attr('y', 0)
        .attr('alignment-baseline', 'middle')
        .text('1');

      legend.append('text')
        .attr('x', legendWidth + 5)
        .attr('y', legendHeight)
        .attr('alignment-baseline', 'middle')
        .text('0');
    }
  }, [data]);

  return (
    <div>
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default D3Visualization;
