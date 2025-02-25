package com.izmir.transportation;

import java.awt.Color;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.geotools.data.DataUtilities;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.JTSFactoryFinder;
import org.geotools.map.FeatureLayer;
import org.geotools.map.Layer;
import org.geotools.map.MapContent;
import org.geotools.referencing.crs.DefaultGeographicCRS;
import org.geotools.styling.SLD;
import org.geotools.styling.Style;
import org.geotools.swing.JMapFrame;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleWeightedGraph;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.Point;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;

import com.izmir.transportation.helper.Edge;
import com.izmir.transportation.helper.Node;

/**
 * A class that models the Izmir transportation network as a weighted graph structure.
 * This class integrates with IzmirBayGraph's point generation and CreateRoadNetwork's path calculations
 * to create a comprehensive graph representation of the transportation network.
 * 
 * The graph uses JGraphT's SimpleWeightedGraph with Point vertices and DefaultWeightedEdge edges,
 * where edge weights represent actual path distances calculated from the road network.
 * 
 * @author yagizugurveren
 */
public class TransportationGraph {
    private final Graph<Node, DefaultWeightedEdge> graph;
    private final Map<Point, Node> pointToNode;
    private final GeometryFactory geometryFactory;
    private final Map<DefaultWeightedEdge, Edge> edgeMap;
    private double maxDistance = 0.0;
    private static final double MIN_EDGE_WIDTH = 1.0;
    private static final double MAX_EDGE_WIDTH = 5.0;
    private AffinityMatrix affinityMatrix;

    /**
     * Constructs a new TransportationGraph with the given points.
     * 
     * @param originalPoints List of points generated by IzmirBayGraph
     */
    public TransportationGraph(List<Point> originalPoints) {
        this.graph = new SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        this.pointToNode = new ConcurrentHashMap<>();
        this.edgeMap = new ConcurrentHashMap<>();
        this.geometryFactory = JTSFactoryFinder.getGeometryFactory();

        for (Point point : originalPoints) {
            Node node = new Node(String.valueOf(point.hashCode()), point);
            graph.addVertex(node);
            pointToNode.put(point, node);
        }
    }

    /**
     * Updates the mapping between an original point and its corresponding network node.
     * 
     * @param originalPoint The original point from IzmirBayGraph
     * @param networkPoint The corresponding point in the road network
     */
    public void updateNodeMapping(Point originalPoint, Point networkPoint) {
        Node networkNode = new Node(String.valueOf(networkPoint.hashCode()), networkPoint);

        if (!graph.containsVertex(networkNode)) {
            graph.addVertex(networkNode);
        }
        pointToNode.put(originalPoint, networkNode);
    }

    /**
     * Adds a weighted connection between two points in the graph.
     * 
     * @param source The source point
     * @param target The target point
     * @param distance The distance of the connection in meters
     * @return The created edge, or null if the connection couldn't be made
     */
    public synchronized DefaultWeightedEdge addConnection(Point source, Point target, double distance) {
        if (source == null || target == null || distance <= 0) {
            return null;
        }

        Node sourceNode = pointToNode.get(source);
        Node targetNode = pointToNode.get(target);

        if (sourceNode == null || targetNode == null) {
            return null;
        }

        synchronized (graph) {
            if (!graph.containsVertex(sourceNode)) {
                graph.addVertex(sourceNode);
            }
            if (!graph.containsVertex(targetNode)) {
                graph.addVertex(targetNode);
            }

            DefaultWeightedEdge graphEdge = graph.addEdge(sourceNode, targetNode);
            if (graphEdge != null) {
                synchronized (this) {
                    maxDistance = Math.max(maxDistance, distance);

                    LineString geometry = geometryFactory.createLineString(new Coordinate[]{
                        source.getCoordinate(),
                        target.getCoordinate()
                    });

                    Edge edge = new Edge(
                        String.valueOf(graphEdge.hashCode()),
                        sourceNode,
                        targetNode,
                        geometry,
                        distance
                    );

                    edgeMap.put(graphEdge, edge);
                    graph.setEdgeWeight(graphEdge, distance);
                }
            }
            return graphEdge;
        }
    }

    /**
     * Updates the weights of all edges in the graph based on the current maximum distance.
     */
    private synchronized void updateEdgeWeights() {
        double localMaxDistance = maxDistance;
        for (Map.Entry<DefaultWeightedEdge, Edge> entry : edgeMap.entrySet()) {
            Edge edge = entry.getValue();
            edge.normalizeWeight(localMaxDistance);
            graph.setEdgeWeight(entry.getKey(), edge.getNormalizedWeight());
        }
    }

    /**
     * Creates and displays a visualization of the transportation network graph.
     * The visualization shows nodes as red circles and edges as blue lines with
     * thickness proportional to their weights.
     */
    public void visualizeGraph() {
        try {
            // Update all edge weights before visualization
            updateEdgeWeights();

            // Create the visualization on the Event Dispatch Thread
            SwingUtilities.invokeLater(() -> {
                try {
                    SimpleFeatureTypeBuilder pointBuilder = new SimpleFeatureTypeBuilder();
                    pointBuilder.setName("Nodes");
                    pointBuilder.setCRS(DefaultGeographicCRS.WGS84);
                    pointBuilder.add("geometry", Point.class);
                    pointBuilder.add("id", String.class);
                    SimpleFeatureType pointType = pointBuilder.buildFeatureType();

                    SimpleFeatureTypeBuilder lineBuilder = new SimpleFeatureTypeBuilder();
                    lineBuilder.setName("Edges");
                    lineBuilder.setCRS(DefaultGeographicCRS.WGS84);
                    lineBuilder.add("geometry", LineString.class);
                    lineBuilder.add("weight", Double.class);
                    lineBuilder.add("label", String.class);
                    SimpleFeatureType lineType = lineBuilder.buildFeatureType();

                    DefaultFeatureCollection nodes = new DefaultFeatureCollection();
                    DefaultFeatureCollection edges = new DefaultFeatureCollection();
                    DefaultFeatureCollection pathCollection = new DefaultFeatureCollection();

                    SimpleFeatureBuilder pointFeatureBuilder = new SimpleFeatureBuilder(pointType);
                    for (Node node : graph.vertexSet()) {
                        pointFeatureBuilder.add(node.getLocation());
                        pointFeatureBuilder.add(node.getId());
                        SimpleFeature feature = pointFeatureBuilder.buildFeature(null);
                        nodes.add(feature);
                    }

                    SimpleFeatureBuilder lineBuilder2 = new SimpleFeatureBuilder(lineType);
                    for (Map.Entry<DefaultWeightedEdge, Edge> entry : edgeMap.entrySet()) {
                        Edge edge = entry.getValue();
                        double weight = edge.getNormalizedWeight();

                        lineBuilder2.add(edge.getGeometry());
                        lineBuilder2.add(weight);
                        lineBuilder2.add(String.format("%.3f", weight));
                        SimpleFeature feature = lineBuilder2.buildFeature(null);
                        edges.add(feature);
                    }

                    Style nodeStyle = SLD.createPointStyle("circle", Color.BLACK, Color.RED, 1.0f, 7);
                    
                    Layer nodesLayer = new FeatureLayer(nodes, nodeStyle);
                    
                    MapContent map = new MapContent();
                    map.setTitle("Izmir Transportation Network Graph");
                    
                    for (Map.Entry<DefaultWeightedEdge, Edge> entry : edgeMap.entrySet()) {
                        Edge edge = entry.getValue();
                        double weight = edge.getNormalizedWeight();
                        
                        float lineWidth = (float) (MIN_EDGE_WIDTH + 
                            weight * (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH));
                        
                        SimpleFeatureType edgeType = DataUtilities.createType("Edge",
                            "geometry:LineString,weight:Double,label:String");
                        SimpleFeatureBuilder edgeBuilder = new SimpleFeatureBuilder(edgeType);
                        
                        edgeBuilder.add(edge.getGeometry());
                        edgeBuilder.add(weight);
                        edgeBuilder.add(String.format("%.3f", weight));
                        
                        DefaultFeatureCollection edgeCollection = new DefaultFeatureCollection();
                        edgeCollection.add(edgeBuilder.buildFeature(null));
                        
                        org.geotools.styling.StyleBuilder styleBuilder = new org.geotools.styling.StyleBuilder();
                        
                        org.geotools.styling.LineSymbolizer lineSymbolizer = styleBuilder.createLineSymbolizer(Color.LIGHT_GRAY, lineWidth);
                        
                        org.geotools.styling.TextSymbolizer textSymbolizer = styleBuilder.createTextSymbolizer();
                        textSymbolizer.setLabel(styleBuilder.attributeExpression("label"));
                        textSymbolizer.setFill(styleBuilder.createFill(Color.BLACK));
                        textSymbolizer.setFont(styleBuilder.createFont("Arial", 12));
                        
                        org.geotools.styling.Rule rule = styleBuilder.createRule(new org.geotools.styling.Symbolizer[]{
                            lineSymbolizer,
                            textSymbolizer
                        });
                        org.geotools.styling.FeatureTypeStyle fts = styleBuilder.createFeatureTypeStyle("Edge", rule);
                        Style edgeStyle = styleBuilder.createStyle();
                        edgeStyle.featureTypeStyles().add(fts);
                        
                        Layer edgeLayer = new FeatureLayer(edgeCollection, edgeStyle);
                        map.addLayer(edgeLayer);
                    }
                    
                    map.addLayer(nodesLayer);

                    JMapFrame mapFrame = new JMapFrame(map);
                    mapFrame.enableToolBar(true);
                    mapFrame.enableStatusBar(true);
                    mapFrame.setSize(800, 600);
                    mapFrame.setLocationRelativeTo(null);
                    mapFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                    mapFrame.setVisible(true);
                } catch (Exception e) {
                    System.err.println("Error in graph visualization: " + e.getMessage());
                    e.printStackTrace();
                }
            });

        } catch (Exception e) {
            System.err.println("Error preparing graph visualization: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Creates and saves the affinity matrix for the transportation network.
     * The matrix will be saved as a CSV file and then displayed in a grid format.
     */
    public void createAffinityMatrix() {
        System.out.println("Creating affinity matrix...");
        affinityMatrix = new AffinityMatrix(graph, edgeMap);
        
        try {
            // Save matrix data as CSV
            String csvFile = "affinity_matrix.csv";
            affinityMatrix.saveToCSV(csvFile);
            System.out.println("Affinity matrix saved to CSV file");
            
            // Display the CSV file in a grid and wait for it to appear
            //System.out.println("Opening affinity matrix visualization...");
            //AffinityMatrix.displayCSV(csvFile);
            
            System.out.println("Affinity matrix has been created and saved to: " + csvFile);
        } catch (IOException e) {
            System.err.println("Error creating affinity matrix: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Gets the affinity matrix if it has been created.
     *
     * @return The affinity matrix, or null if it hasn't been created yet
     */
    public AffinityMatrix getAffinityMatrix() {
        return affinityMatrix;
    }

    /**
     * Gets the underlying JGraphT graph structure.
     * 
     * @return The graph object
     */
    public Graph<Node, DefaultWeightedEdge> getGraph() {
        return graph;
    }

    /**
     * Gets the mapping between points and their corresponding nodes.
     * 
     * @return The point to node mapping
     */
    public Map<Point, Node> getPointToNode() {
        return pointToNode;
    }

    /**
     * Gets the mapping between graph edges and our custom Edge objects.
     * 
     * @return The edge mapping
     */
    public Map<DefaultWeightedEdge, Edge> getEdgeMap() {
        return edgeMap;
    }
} 