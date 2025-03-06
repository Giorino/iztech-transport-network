package com.izmir.transportation.persistence;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LineString;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Jackson deserializer for JTS LineString geometry objects.
 */
public class LineStringDeserializer extends JsonDeserializer<LineString> {
    private static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory();

    @Override
    public LineString deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
        JsonNode node = p.getCodec().readTree(p);
        JsonNode coordsNode = node.get("coordinates");
        
        List<Coordinate> coordinates = new ArrayList<>();
        for (JsonNode coordNode : coordsNode) {
            double x = coordNode.get(0).asDouble();
            double y = coordNode.get(1).asDouble();
            coordinates.add(new Coordinate(x, y));
        }
        
        return GEOMETRY_FACTORY.createLineString(coordinates.toArray(new Coordinate[0]));
    }
} 