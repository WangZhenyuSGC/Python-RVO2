/*
 * Agent.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#include "Agent.h"

#include "KdTree.h"
#include "Obstacle.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <iostream>

namespace RVO {
    Agent::Agent(RVOSimulator *sim) : 
    maxNeighbors_(0), maxSpeed_(0.0f), neighborDist_(0.0f), radius_(0.0f), sim_(sim), timeHorizon_(0.0f), timeHorizonObst_(0.0f), 
    collabCoeff_(0.5f), isDeadLock_(false), 
    minErrorHolo_(0.0f),maxErrorHolo_(0.0f), velMaxW_(0.0f), wMax_(0.0f), curAllowedError_(0.0f), timeToHolo_(0.0f), 
    heading_(0.0f), angVel_(0.0f), wheelBase_(0.0f), isUsingNH_(false),
    id_(0)
    {}

    void Agent::computeNeighbors()
    {
        obstacleNeighbors_.clear();
        float rangeSq = sqr(timeHorizonObst_ * maxSpeed_ + radius_);
        sim_->kdTree_->computeObstacleNeighbors(this, rangeSq);

        agentNeighbors_.clear();

        if (maxNeighbors_ > 0) {
            rangeSq = sqr(neighborDist_);
            sim_->kdTree_->computeAgentNeighbors(this, rangeSq);
        }
    }

    /* Search for the best new velocity. */
    void Agent::computeNewVelocity()
    {
        orcaLines_.clear();

        const float invTimeHorizonObst = 1.0f / timeHorizonObst_;

        // Non-holonomic constraints追加 1017
        // まずは一番近い障害物との距離を計算
        float minDistObst = std::numeric_limits<float>::infinity();

        /* Create obstacle ORCA lines. */
        for (size_t i = 0; i < obstacleNeighbors_.size(); ++i) {

            const Obstacle *obstacle1 = obstacleNeighbors_[i].second;
            const Obstacle *obstacle2 = obstacle1->nextObstacle_;

            const Vector2 relativePosition1 = obstacle1->point_ - position_;
            const Vector2 relativePosition2 = obstacle2->point_ - position_;
            
            const float dist = distSqPointLineSegment(obstacle1->point_, obstacle2->point_, position_);
            if (dist < minDistObst){
                minDistObst = dist;
            }
            /*
             * Check if velocity obstacle of obstacle is already taken care of by
             * previously constructed obstacle ORCA lines.
             */
            bool alreadyCovered = false;

            for (size_t j = 0; j < orcaLines_.size(); ++j) {
                if (det(invTimeHorizonObst * relativePosition1 - orcaLines_[j].point, orcaLines_[j].direction) - invTimeHorizonObst * radius_ >= -RVO_EPSILON && det(invTimeHorizonObst * relativePosition2 - orcaLines_[j].point, orcaLines_[j].direction) - invTimeHorizonObst * radius_ >=  -RVO_EPSILON) {
                    alreadyCovered = true;
                    break;
                }
            }

            if (alreadyCovered) {
                continue;
            }

            /* Not yet covered. Check for collisions. */

            const float distSq1 = absSq(relativePosition1);
            const float distSq2 = absSq(relativePosition2);

            const float radiusSq = sqr(radius_);

            const Vector2 obstacleVector = obstacle2->point_ - obstacle1->point_;
            const float s = (-relativePosition1 * obstacleVector) / absSq(obstacleVector);
            const float distSqLine = absSq(-relativePosition1 - s * obstacleVector);

            Line line;

            if (s < 0.0f && distSq1 <= radiusSq) {
                /* Collision with left vertex. Ignore if non-convex. */
                if (obstacle1->isConvex_) {
                    line.point = Vector2(0.0f, 0.0f);
                    line.direction = normalize(Vector2(-relativePosition1.y(), relativePosition1.x()));
                    orcaLines_.push_back(line);
                }

                continue;
            }
            else if (s > 1.0f && distSq2 <= radiusSq) {
                /* Collision with right vertex. Ignore if non-convex
                 * or if it will be taken care of by neighoring obstace */
                if (obstacle2->isConvex_ && det(relativePosition2, obstacle2->unitDir_) >= 0.0f) {
                    line.point = Vector2(0.0f, 0.0f);
                    line.direction = normalize(Vector2(-relativePosition2.y(), relativePosition2.x()));
                    orcaLines_.push_back(line);
                }

                continue;
            }
            else if (s >= 0.0f && s < 1.0f && distSqLine <= radiusSq) {
                /* Collision with obstacle segment. */
                line.point = Vector2(0.0f, 0.0f);
                line.direction = -obstacle1->unitDir_;
                orcaLines_.push_back(line);
                continue;
            }

            /*
             * No collision.
             * Compute legs. When obliquely viewed, both legs can come from a single
             * vertex. Legs extend cut-off line when nonconvex vertex.
             */

            Vector2 leftLegDirection, rightLegDirection;

            if (s < 0.0f && distSqLine <= radiusSq) {
                /*
                 * Obstacle viewed obliquely so that left vertex
                 * defines velocity obstacle.
                 */
                if (!obstacle1->isConvex_) {
                    /* Ignore obstacle. */
                    continue;
                }

                obstacle2 = obstacle1;

                const float leg1 = std::sqrt(distSq1 - radiusSq);
                leftLegDirection = Vector2(relativePosition1.x() * leg1 - relativePosition1.y() * radius_, relativePosition1.x() * radius_ + relativePosition1.y() * leg1) / distSq1;
                rightLegDirection = Vector2(relativePosition1.x() * leg1 + relativePosition1.y() * radius_, -relativePosition1.x() * radius_ + relativePosition1.y() * leg1) / distSq1;
            }
            else if (s > 1.0f && distSqLine <= radiusSq) {
                /*
                 * Obstacle viewed obliquely so that
                 * right vertex defines velocity obstacle.
                 */
                if (!obstacle2->isConvex_) {
                    /* Ignore obstacle. */
                    continue;
                }

                obstacle1 = obstacle2;

                const float leg2 = std::sqrt(distSq2 - radiusSq);
                leftLegDirection = Vector2(relativePosition2.x() * leg2 - relativePosition2.y() * radius_, relativePosition2.x() * radius_ + relativePosition2.y() * leg2) / distSq2;
                rightLegDirection = Vector2(relativePosition2.x() * leg2 + relativePosition2.y() * radius_, -relativePosition2.x() * radius_ + relativePosition2.y() * leg2) / distSq2;
            }
            else {
                /* Usual situation. */
                if (obstacle1->isConvex_) {
                    const float leg1 = std::sqrt(distSq1 - radiusSq);
                    leftLegDirection = Vector2(relativePosition1.x() * leg1 - relativePosition1.y() * radius_, relativePosition1.x() * radius_ + relativePosition1.y() * leg1) / distSq1;
                }
                else {
                    /* Left vertex non-convex; left leg extends cut-off line. */
                    leftLegDirection = -obstacle1->unitDir_;
                }

                if (obstacle2->isConvex_) {
                    const float leg2 = std::sqrt(distSq2 - radiusSq);
                    rightLegDirection = Vector2(relativePosition2.x() * leg2 + relativePosition2.y() * radius_, -relativePosition2.x() * radius_ + relativePosition2.y() * leg2) / distSq2;
                }
                else {
                    /* Right vertex non-convex; right leg extends cut-off line. */
                    rightLegDirection = obstacle1->unitDir_;
                }
            }

            /*
             * Legs can never point into neighboring edge when convex vertex,
             * take cutoff-line of neighboring edge instead. If velocity projected on
             * "foreign" leg, no constraint is added.
             */

            const Obstacle *const leftNeighbor = obstacle1->prevObstacle_;

            bool isLeftLegForeign = false;
            bool isRightLegForeign = false;

            if (obstacle1->isConvex_ && det(leftLegDirection, -leftNeighbor->unitDir_) >= 0.0f) {
                /* Left leg points into obstacle. */
                leftLegDirection = -leftNeighbor->unitDir_;
                isLeftLegForeign = true;
            }

            if (obstacle2->isConvex_ && det(rightLegDirection, obstacle2->unitDir_) <= 0.0f) {
                /* Right leg points into obstacle. */
                rightLegDirection = obstacle2->unitDir_;
                isRightLegForeign = true;
            }

            /* Compute cut-off centers. */
            const Vector2 leftCutoff = invTimeHorizonObst * (obstacle1->point_ - position_);
            const Vector2 rightCutoff = invTimeHorizonObst * (obstacle2->point_ - position_);
            const Vector2 cutoffVec = rightCutoff - leftCutoff;

            /* Project current velocity on velocity obstacle. */

            /* Check if current velocity is projected on cutoff circles. */
            const float t = (obstacle1 == obstacle2 ? 0.5f : ((velocity_ - leftCutoff) * cutoffVec) / absSq(cutoffVec));
            const float tLeft = ((velocity_ - leftCutoff) * leftLegDirection);
            const float tRight = ((velocity_ - rightCutoff) * rightLegDirection);

            if ((t < 0.0f && tLeft < 0.0f) || (obstacle1 == obstacle2 && tLeft < 0.0f && tRight < 0.0f)) {
                /* Project on left cut-off circle. */
                const Vector2 unitW = normalize(velocity_ - leftCutoff);

                line.direction = Vector2(unitW.y(), -unitW.x());
                line.point = leftCutoff + radius_ * invTimeHorizonObst * unitW;
                orcaLines_.push_back(line);
                continue;
            }
            else if (t > 1.0f && tRight < 0.0f) {
                /* Project on right cut-off circle. */
                const Vector2 unitW = normalize(velocity_ - rightCutoff);

                line.direction = Vector2(unitW.y(), -unitW.x());
                line.point = rightCutoff + radius_ * invTimeHorizonObst * unitW;
                orcaLines_.push_back(line);
                continue;
            }

            /*
             * Project on left leg, right leg, or cut-off line, whichever is closest
             * to velocity.
             */
            const float distSqCutoff = ((t < 0.0f || t > 1.0f || obstacle1 == obstacle2) ? std::numeric_limits<float>::infinity() : absSq(velocity_ - (leftCutoff + t * cutoffVec)));
            const float distSqLeft = ((tLeft < 0.0f) ? std::numeric_limits<float>::infinity() : absSq(velocity_ - (leftCutoff + tLeft * leftLegDirection)));
            const float distSqRight = ((tRight < 0.0f) ? std::numeric_limits<float>::infinity() : absSq(velocity_ - (rightCutoff + tRight * rightLegDirection)));

            if (distSqCutoff <= distSqLeft && distSqCutoff <= distSqRight) {
                /* Project on cut-off line. */
                line.direction = -obstacle1->unitDir_;
                line.point = leftCutoff + radius_ * invTimeHorizonObst * Vector2(-line.direction.y(), line.direction.x());
                orcaLines_.push_back(line);
                continue;
            }
            else if (distSqLeft <= distSqRight) {
                /* Project on left leg. */
                if (isLeftLegForeign) {
                    continue;
                }

                line.direction = leftLegDirection;
                line.point = leftCutoff + radius_ * invTimeHorizonObst * Vector2(-line.direction.y(), line.direction.x());
                orcaLines_.push_back(line);
                continue;
            }
            else {
                /* Project on right leg. */
                if (isRightLegForeign) {
                    continue;
                }

                line.direction = -rightLegDirection;
                line.point = rightCutoff + radius_ * invTimeHorizonObst * Vector2(-line.direction.y(), line.direction.x());
                orcaLines_.push_back(line);
                continue;
            }
        }

        const size_t numObstLines = orcaLines_.size();

        const float invTimeHorizon = 1.0f / timeHorizon_;

        float minDistAgent = std::numeric_limits<float>::infinity();

        if (collabCoeff_ == 0.){
            // Agent doesn't care about other agents ==> just avoid the statics @ prefVelocity_
        }

        else if (collabCoeff_ > 0.){
            // Agent is collaborative (trying to avoid collisions)

            // 一番近いAgentとの距離を計算

            /* Create agent ORCA lines. */
            for (size_t i = 0; i < agentNeighbors_.size(); ++i) {
                const Agent *const other = agentNeighbors_[i].second;

                const Vector2 relativePosition = other->position_ - position_;
                const Vector2 relativeVelocity = velocity_ - other->velocity_; // might as well compute VO based on intended relVel, not a relVel that's not desirable?
                const float distSq = absSq(relativePosition);
                float combinedRadius = radius_ + other->radius_;
                if (isUsingNH_) {
                    combinedRadius = radius_ + other->radius_ + curAllowedError_;
                } // 1022: NH用

                const float combinedRadiusSq = sqr(combinedRadius);

                if (distSq < minDistAgent){
                   minDistAgent = distSq;
                }
                
                Line line;
                Vector2 u;

                if (distSq > combinedRadiusSq) {
                    /* No collision. */
                    const Vector2 w = relativeVelocity - invTimeHorizon * relativePosition;
                    /* Vector from cutoff center to relative velocity. */
                    const float wLengthSq = absSq(w);

                    const float dotProduct1 = w * relativePosition;

                    if (dotProduct1 < 0.0f && sqr(dotProduct1) > combinedRadiusSq * wLengthSq) {
                        /* Project on cut-off circle. */
                        const float wLength = std::sqrt(wLengthSq);
                        const Vector2 unitW = w / wLength;

                        line.direction = Vector2(unitW.y(), -unitW.x());
                        u = (combinedRadius * invTimeHorizon - wLength) * unitW;
                    }
                    else {
                        /* Project on legs. */
                        const float leg = std::sqrt(distSq - combinedRadiusSq);

                        if (det(relativePosition, w) > 0.0f) {
                            /* Project on left leg. */
                            line.direction = Vector2(relativePosition.x() * leg - relativePosition.y() * combinedRadius, relativePosition.x() * combinedRadius + relativePosition.y() * leg) / distSq;
                        }
                        else {
                            /* Project on right leg. */
                            line.direction = -Vector2(relativePosition.x() * leg + relativePosition.y() * combinedRadius, -relativePosition.x() * combinedRadius + relativePosition.y() * leg) / distSq;
                        }

                        const float dotProduct2 = relativeVelocity * line.direction;

                        u = dotProduct2 * line.direction - relativeVelocity;
                    }
                }
                else {
                    // /* Debug print for time step issue*/
                    // std::cout << "***Detected collision, using timeStep to calculate velocity***" << std::endl;
                    /* Collision. Project on cut-off circle of time timeStep. */
                    const float invTimeStep = 1.0f / sim_->timeStep_;

                    /* Vector from cutoff center to relative velocity. */
                    const Vector2 w = relativeVelocity - invTimeStep * relativePosition;

                    const float wLength = abs(w);
                    const Vector2 unitW = w / wLength;

                    line.direction = Vector2(unitW.y(), -unitW.x());
                    u = (combinedRadius * invTimeStep - wLength) * unitW;
                    // std::cout << "Collision agent id: " << id_ << std::endl;
                    // std::cout << "Collision agent position: " << position_ << std::endl;
                    // std::cout << "Collision agent velocity: " << velocity_ << std::endl;
                    // std::cout << "Collision relative velocity: " << relativeVelocity << std::endl;
                    // std::cout << "Collision relative position: " << relativePosition << std::endl;
                    // std::cout<< "calculated w: " << w << std::endl;
                    // std::cout << "combinedRadius * invTimeStep - wLength: " << combinedRadius * invTimeStep - wLength << std::endl;
                    // std::cout << "calculated u: " << u << std::endl;
                    // std::cout << " " << std::endl;
                }

                // MFE changed velocity_ to prefVelocity_ s.t. ego agent adjusts its preferred speed, not current speed
                // This corresponds to choice 2 in the ORCA paper section 5.2
                //  line.point = prefVelocity_  + collabCoeff_ * u;
                line.point = velocity_  + collabCoeff_ * u;
                // line.point = velocity_  + 0.5 * u;
                orcaLines_.push_back(line);
            }
        }

        else{
            // Agent has negative collaboration ==> actively trying to crash into someone
            if (agentNeighbors_.size() == 0){
                // nobody is nearby, just go prefVelocity_ w/o any constraints
            }
            else{
                float minDistSq = std::numeric_limits<float>::infinity();
                size_t closestAgentIndex;
                for (size_t i = 0; i < agentNeighbors_.size(); ++i) {
                    const Agent *const other = agentNeighbors_[i].second;
                    float distSq = absSq(other->position_ - position_);
                    if (distSq < minDistSq){
                        minDistSq = distSq;
                        closestAgentIndex = i;
                    }
                }
                const Agent *const other = agentNeighbors_[closestAgentIndex].second;
                const Vector2 relativePosition = other->position_ - position_;


                // // Method 1: Move along vector in direction of vb+(pb-pa), normalized to maxspeed, scaled by -collabCoeff
                // // Issue: Choosing a low collabCoeff (close to 0) probably will make the chosen direction not actually enter the VO  
                // prefVelocity_ = normalize(other->velocity_+relativePosition)*(-std::max(-1.f, collabCoeff_))*maxSpeed_;

                // Method 2: Add two linear constraints for the two legs of the VO, but facing inward so the optimizer
                // chooses a velocity inside the VO. prefVelocity_ is set to along the vb to vb+(pb-pa) ray, scaled along the (pb-pa) addition by collabCoeff
                // This prefVelocity_ corresponds to trying to hit the other agent's center in -1/collabCoeff seconds (more negative collabCoeff == sooner collision)
                // so the collabCoeff trades off mimicing the other agent (collab=0 ==> prefVel = other->veloc) vs. trying to hit immediately (collab=-inf ==> prefVel=relPos)
                // ==> if things are "good" (entire VO region of interest lies in maxSpd ball): should just pick prefVelocity_
                // if part of the VO ROI is outside the maxSpd ball around the agent, the optimization gets a little weird (and I haven't throught through it yet)
                const float distSq = absSq(relativePosition);
                const float combinedRadius = radius_ + other->radius_;
                const float combinedRadiusSq = sqr(combinedRadius);
                const float leg = std::sqrt(distSq - combinedRadiusSq);
                Line line;
                /* left leg. */
                line.direction = -Vector2(relativePosition.x() * leg - relativePosition.y() * combinedRadius, relativePosition.x() * combinedRadius + relativePosition.y() * leg) / distSq;
                line.point = other->velocity_ + relativePosition;
                orcaLines_.push_back(line);
                /* right leg. */
                line.direction = Vector2(relativePosition.x() * leg + relativePosition.y() * combinedRadius, -relativePosition.x() * combinedRadius + relativePosition.y() * leg) / distSq;
                line.point = other->velocity_ + relativePosition;
                orcaLines_.push_back(line);

                prefVelocity_ = other->velocity_ + (-collabCoeff_)*normalize(relativePosition);

            }
        }

        // 線形計画での計算に入る前に、Non-holonomic用の制約を追加
        double minDist = sqrt(std::min(minDistAgent, minDistObst));
        additionalOrcaLines_.clear();
        if (isUsingNH_) {
            addNHConstraints(minDist);
        }

        // 追加された制約はadditionalOrcaLines_に格納されているため、計算に移る前にinsertする
        orcaLines_.insert(orcaLines_.end(), additionalOrcaLines_.begin(), additionalOrcaLines_.end());

        size_t lineFail = linearProgram2(orcaLines_, maxSpeed_, prefVelocity_, false, newVelocity_);

        if (lineFail < orcaLines_.size() || isDeadLock_) 
        {
            // DeadLockの場合は優先的にlinearProgram3で解を求める
            linearProgram3(orcaLines_, numObstLines, lineFail, maxSpeed_, newVelocity_);
        }
    }

    void Agent::insertAgentNeighbor(const Agent *agent, float &rangeSq)
    {
        if (this != agent) {
            const float distSq = absSq(position_ - agent->position_);

            if (distSq < rangeSq) {
                if (agentNeighbors_.size() < maxNeighbors_) {
                    agentNeighbors_.push_back(std::make_pair(distSq, agent));
                }

                size_t i = agentNeighbors_.size() - 1;

                while (i != 0 && distSq < agentNeighbors_[i - 1].first) {
                    agentNeighbors_[i] = agentNeighbors_[i - 1];
                    --i;
                }

                agentNeighbors_[i] = std::make_pair(distSq, agent);

                if (agentNeighbors_.size() == maxNeighbors_) {
                    rangeSq = agentNeighbors_.back().first;
                }
            }
        }
    }

    void Agent::insertObstacleNeighbor(const Obstacle *obstacle, float rangeSq)
    {
        const Obstacle *const nextObstacle = obstacle->nextObstacle_;

        const float distSq = distSqPointLineSegment(obstacle->point_, nextObstacle->point_, position_);

        if (distSq < rangeSq) {
            obstacleNeighbors_.push_back(std::make_pair(distSq, obstacle));

            size_t i = obstacleNeighbors_.size() - 1;

            while (i != 0 && distSq < obstacleNeighbors_[i - 1].first) {
                obstacleNeighbors_[i] = obstacleNeighbors_[i - 1];
                --i;
            }

            obstacleNeighbors_[i] = std::make_pair(distSq, obstacle);
        }
    }

    void Agent::update()
    {
        // std::cout << newVelocity_ << std::endl;
        velocity_ = newVelocity_;
        position_ += velocity_ * sim_->timeStep_;
    }

    bool linearProgram1(const std::vector<Line> &lines, size_t lineNo, float radius, const Vector2 &optVelocity, bool directionOpt, Vector2 &result)
    {
        const float dotProduct = lines[lineNo].point * lines[lineNo].direction;
        const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines[lineNo].point);

        if (discriminant < 0.0f) {
            /* Max speed circle fully invalidates line lineNo. */
            return false;
        }

        const float sqrtDiscriminant = std::sqrt(discriminant);
        float tLeft = -dotProduct - sqrtDiscriminant;
        float tRight = -dotProduct + sqrtDiscriminant;

        for (size_t i = 0; i < lineNo; ++i) {
            const float denominator = det(lines[lineNo].direction, lines[i].direction);
            const float numerator = det(lines[i].direction, lines[lineNo].point - lines[i].point);

            if (std::fabs(denominator) <= RVO_EPSILON) {
                /* Lines lineNo and i are (almost) parallel. */
                if (numerator < 0.0f) {
                    return false;
                }
                else {
                    continue;
                }
            }

            const float t = numerator / denominator;

            if (denominator >= 0.0f) {
                /* Line i bounds line lineNo on the right. */
                tRight = std::min(tRight, t);
            }
            else {
                /* Line i bounds line lineNo on the left. */
                tLeft = std::max(tLeft, t);
            }

            if (tLeft > tRight) {
                return false;
            }
        }

        if (directionOpt) {
            /* Optimize direction. */
            if (optVelocity * lines[lineNo].direction > 0.0f) {
                /* Take right extreme. */
                result = lines[lineNo].point + tRight * lines[lineNo].direction;
            }
            else {
                /* Take left extreme. */
                result = lines[lineNo].point + tLeft * lines[lineNo].direction;
            }
        }
        else {
            /* Optimize closest point. */
            const float t = lines[lineNo].direction * (optVelocity - lines[lineNo].point);

            if (t < tLeft) {
                result = lines[lineNo].point + tLeft * lines[lineNo].direction;
            }
            else if (t > tRight) {
                result = lines[lineNo].point + tRight * lines[lineNo].direction;
            }
            else {
                result = lines[lineNo].point + t * lines[lineNo].direction;
            }
        }

        return true;
    }

    size_t linearProgram2(const std::vector<Line> &lines, float radius, const Vector2 &optVelocity, bool directionOpt, Vector2 &result)
    {
        if (directionOpt) {
            /*
             * Optimize direction. Note that the optimization velocity is of unit
             * length in this case.
             */
            result = optVelocity * radius;
        }
        else if (absSq(optVelocity) > sqr(radius)) {
            /* Optimize closest point and outside circle. */
            result = normalize(optVelocity) * radius;
        }
        else {
            /* Optimize closest point and inside circle. */
            result = optVelocity;
        }

        for (size_t i = 0; i < lines.size(); ++i) {
            if (det(lines[i].direction, lines[i].point - result) > 0.0f) {
                /* Result does not satisfy constraint i. Compute new optimal result. */
                const Vector2 tempResult = result;

                if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, result)) {
                    result = tempResult;
                    return i;
                }
            }
        }

        return lines.size();
    }

    void linearProgram3(const std::vector<Line> &lines, size_t numObstLines, size_t beginLine, float radius, Vector2 &result)
    {
        float distance = 0.0f;

        for (size_t i = beginLine; i < lines.size(); ++i) {
            if (det(lines[i].direction, lines[i].point - result) > distance) {
                /* Result does not satisfy constraint of line i. */
                std::vector<Line> projLines(lines.begin(), lines.begin() + static_cast<ptrdiff_t>(numObstLines));

                for (size_t j = numObstLines; j < i; ++j) {
                    Line line;

                    float determinant = det(lines[i].direction, lines[j].direction);

                    if (std::fabs(determinant) <= RVO_EPSILON) {
                        /* Line i and line j are parallel. */
                        if (lines[i].direction * lines[j].direction > 0.0f) {
                            /* Line i and line j point in the same direction. */
                            continue;
                        }
                        else {
                            /* Line i and line j point in opposite direction. */
                            line.point = 0.5f * (lines[i].point + lines[j].point);
                        }
                    }
                    else {
                        line.point = lines[i].point + (det(lines[j].direction, lines[i].point - lines[j].point) / determinant) * lines[i].direction;
                    }

                    line.direction = normalize(lines[j].direction - lines[i].direction);
                    projLines.push_back(line);
                }

                const Vector2 tempResult = result;

                if (linearProgram2(projLines, radius, Vector2(-lines[i].direction.y(), lines[i].direction.x()), true, result) < projLines.size()) {
                    /* This should in principle not happen.  The result is by definition
                     * already in the feasible region of this linear program. If it fails,
                     * it is due to small floating point error, and the current result is
                     * kept.
                     */
                    result = tempResult;
                }

                distance = det(lines[i].direction, lines[i].point - result);
            }
        }
    }

    void Agent::addNHConstraints(double min_dist){
        double min_error = minErrorHolo_;
        double max_error = maxErrorHolo_;
        double error = max_error;
        // double v_max_ang = wMax_ * wheelBase_ / 2.0 - std::abs(angVel_) * wheelBase_ / 2.0; //  |v(t)| ≤ vmax,ω = vmax−|ω(t)|·L/2
        double v_max_ang = maxSpeed_;

        // すでに衝突したら
        if (min_dist < 2.0 * radius_) {
            error = (max_error - min_error) / ((2 * radius_) * (2 * radius_)) * min_dist * min_dist
                    + min_error; // もともと使われているsqrは実際a * aなので、ここでも同じように計算
            if (min_dist < 0) {
                error = min_error;
            }
        }

        curAllowedError_ = 1.0 / 3.0 * curAllowedError_ + 2.0 / 3.0 * error;

        double speed_ang = atan2(prefVelocity_.y(), prefVelocity_.x());
        // 今の向きとの差を計算
        double dif_ang = normalizeAngle(speed_ang) - normalizeAngle(heading_); // TODO: Normalize_angleを使って！！！
        double min_theta = normalizeAngle(dif_ang);

        // std::cout << "**************Agent ID:" << id_ << std::endl;

        // 現在の速度と目標速度の差が大きい場合、PI/2の角度で制約を追加
        if (std::abs(dif_ang) > M_PI / 2.0) { // || curAllowedError_ < 2.0 * min_error) {
            // std::cout << "---------------Angle Diff Huge-----------" << std::endl;
            // PI/2の角度なら、横方向に移動する場合、フォローできる速度を計算
            double max_track_speed = calculateMaxTrackSpeedAngle(timeToHolo_, M_PI / 2.0, curAllowedError_,
                                                                 maxSpeed_, wMax_, v_max_ang);
            if (max_track_speed <= 2 * min_error) {
                // std::cout << "Using 2 * min_error instead" << std::endl;
                max_track_speed = 2 * min_error;
            }

            addMovementConstraintsDiffSimple(max_track_speed, heading_, additionalOrcaLines_);
        }
        // else {
        //     std::cout << "---------------Angle Diff Small-----------" << std::endl;
        //     addMovementConstraintsDiff(curAllowedError_, timeToHolo_, maxSpeed_, wMax_, heading_, v_max_ang,
        //                                additionalOrcaLines_);
        // }

        // DEBUG PRINT
        // std::cout << "error: " << error << std::endl;
        // std::cout << "curAllowedError_: " << curAllowedError_ << std::endl;
        // std::cout << "v_max_ang: " << v_max_ang << std::endl;
        // std::cout << "speed_ang: " << speed_ang << std::endl;
        // std::cout << "dif_ang: " << dif_ang << std::endl;
        // std::cout << "heading_: " << heading_ << std::endl;
    } 
    
    void Agent::addMovementConstraintsDiff(double error, double T, double max_vel_x, double max_vel_th, double heading,
                                           double v_max_ang, std::vector<RVO::Line> &additional_orca_lines) {
        double min_theta = M_PI / 2.0;
        double max_track_speed = calculateMaxTrackSpeedAngle(T, min_theta, error, max_vel_x, max_vel_th, v_max_ang);

        RVO::Vector2 first_point = max_track_speed * RVO::Vector2(cos(heading - min_theta), sin(heading - min_theta));

        double steps = 10.0;
        double step_size = -M_PI/ steps;
        // RVO::Line line;
        // line.point = -max_track_speed * Vector2(cos(heading), sin(heading));
        // line.direction = normalize(first_point);
        // additional_orca_lines.push_back(line);

        for (int i = 1; i <= (int) steps; i++) {
            RVO::Line line;
            double cur_ang = min_theta + i * step_size;
            RVO::Vector2 second_point = RVO::Vector2(cos(heading - cur_ang), sin(heading - cur_ang));
            double track_speed = calculateMaxTrackSpeedAngle(T, cur_ang, error, max_vel_x, max_vel_th, v_max_ang);
            second_point = track_speed * second_point;
            line.point = first_point;
            line.direction = normalize(second_point - first_point);
            additional_orca_lines.push_back(line);
            first_point = second_point;
        }
    }  

    void Agent::addMovementConstraintsDiffSimple(double max_track_speed, double heading,
                                            std::vector<RVO::Line> &additional_orca_lines) {
        RVO::Line maxVel1;
        RVO::Line maxVel2;
        
        RVO::Vector2 dir = RVO::Vector2(cos(heading), sin(heading));
        maxVel1.point = max_track_speed * RVO::Vector2(-dir.y(), dir.x());
        maxVel1.direction = -dir;
        maxVel2.direction = dir;
        maxVel2.point = -max_track_speed * RVO::Vector2(-dir.y(), dir.x());
        additional_orca_lines.push_back(maxVel1);
        additional_orca_lines.push_back(maxVel2);
    }
}

// この部分の計算はもう大丈夫はず
double beta(double T, double theta, double v_max_ang) {
    return -((2.0 * T * T * sin(theta)) / theta) * v_max_ang;
}

double gamma(double T, double theta, double error, double v_max_ang) {
    return ((2.0 * T * T * (1.0 - cos(theta))) / (theta * theta)) * v_max_ang * v_max_ang - error * error;
}

double calcVstar(double vh, double theta) {
    return vh * ((theta * sin(theta)) / (2.0 * (1.0 - cos(theta))));
}

double calcVstarError(double T, double theta, double error) {
    return calcVstar(error / T, theta) *
            sqrt((2.0 * (1.0 - cos(theta))) / (2.0 * (1.0 - cos(theta)) - pow(sin(theta), 2)));
}

double calculateMaxTrackSpeedAngle(double T, double theta, double error, double max_vel_x, double max_vel_th,
                                    double v_max_ang) {
    //　角度差が小さい場合、直進速度を返す
    if (fabs(theta) <= EPSILON)
        return max_vel_x;
    if (fabs(theta / T) <= max_vel_th) {
        double vstar_error = calcVstarError(T, theta, error);
        if (vstar_error <= v_max_ang) {
            return std::min(vstar_error * (2.0 * (1.0 - cos(theta))) / (theta * sin(theta)), max_vel_x);
        }
        else {
            double a, b, g;
            a = T * T;
            b = beta(T, theta, v_max_ang);
            g = gamma(T, theta, error, v_max_ang);
            return std::min((-b + sqrt(b * b - 4 * a * g)) / (2.0 * g), max_vel_x);
        }
    }
    else {
        return std::min(sign(theta) * error * max_vel_th / theta, max_vel_x);
    }
}

double sign(double x) {
    return (x > 0) - (x < 0);
}

double normalizeAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2 * M_PI;
    }
    return angle;
}