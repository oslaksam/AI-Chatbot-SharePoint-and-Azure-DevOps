# Release Notes for Version {{buildDetails.buildNumber}} 🎉👍

## Build Information
* **Branch**: {{buildDetails.sourceBranch}}

## Associated Pull Requests ({{pullRequests.length}})
{{#forEach pullRequests}}
* **PR #{{this.pullRequestId}}**: [{{this.title}}]({{replace (replace this.url "_apis/git/repositories" "_git") "pullRequests" "pullRequest"}})
  - **Description**: {{this.description}}
  - **Associated Work Items**
    {{#forEach this.associatedWorkitems}}
      - **WI #{{this.id}}**: {{lookup this.fields 'System.Title'}}
    {{/forEach}}
  - **Associated Commits**
    {{#forEach this.associatedCommits}}
      - Commit [{{this.commitId}}]({{this.remoteUrl}}): {{this.comment}}
    {{/forEach}}
{{/forEach}}

## Work Items ({{workItems.length}})
{{#forEach workItems}}
* **WI #{{this.id}}**: [{{lookup this.fields 'System.Title'}}]({{replace this.url "_apis/wit/workItems" "_workitems/edit"}})
  - **Type**: {{lookup this.fields 'System.WorkItemType'}}
  - **Assigned To**: {{#with (lookup this.fields 'System.AssignedTo')}}{{displayName}}{{/with}}
  - **State**: {{lookup this.fields 'System.State'}}
  - **Description**: {{{lookup this.fields 'System.Description'}}}
{{/forEach}}

## Commits ({{commits.length}})
{{#forEach commits}}
* Commit [{{this.id}}]({{this.remoteUrl}}): {{this.comment}}
  - **Author**: {{this.author.displayName}}
  - **File Changes**: {{this.changes.length}}
{{/forEach}}

    
## Artifacts published by build ({{publishedArtifacts.length}})

| Build ID | Name | Type | Download |
|----------|------|------|----------|
{{#forEach publishedArtifacts}}
| {{this.id}} | {{this.name}} | {{this.resource.type}} | [Download as ZIP]({{this.resource.downloadUrl}}) |
{{/forEach}}
