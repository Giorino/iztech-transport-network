package com.izmir.transportation.persistence;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;

import java.io.IOException;

/**
 * Jackson deserializer for JTS Point geometry objects.
 */
public class PointDeserializer extends JsonDeserializer<Point> {
    private static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory();

    @Override
    public Point deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
        JsonNode node = p.getCodec().readTree(p);
        double x = node.get("x").asDouble();
        double y = node.get("y").asDouble();
        return GEOMETRY_FACTORY.createPoint(new Coordinate(x, y));
    }
} 